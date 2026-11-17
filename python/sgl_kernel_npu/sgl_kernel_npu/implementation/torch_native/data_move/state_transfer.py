"""PyTorch references adapted from SGLang's MambaPoolHost (Apache-2.0).

Source: sgl-project/sglang@6b13469f91d6ce72d64c1760f37475626926c077,
python/sglang/srt/mem_cache/pool_host/mamba.py:
_copy_tensor_pf_lf and _copy_tensor_all_layers_lf_pf.
https://github.com/sgl-project/sglang/blob/6b13469f91d6ce72d64c1760f37475626926c077/python/sglang/srt/mem_cache/pool_host/mamba.py

The gather/scatter references are reused with physical payload views for ACL
copies, whole-page indices for KV, and one call per state component. H2D
updates are staged on CPU, then copied back through the original tensor shape;
this avoids accelerator scatter writes through flattened int8 storage views.
"""

import torch

# Native direction ABI; keep this reference independent of public facade enums.
_D2H = 2


def _payload(tensor):
    stride = 1
    for pitch, size in sorted(
        (s, n) for s, n in zip(tensor.stride(), tensor.shape) if n > 1
    ):
        if pitch != stride:
            raise RuntimeError("slot payload must be physically dense")
        stride *= size
    # ACL copies physically dense payloads in storage order, including transposed states.
    return tensor.as_strided((tensor.numel(),), (1,))


def _slot_pairs(device_indices, host_indices, flags):
    if flags.value & 2 != 2 or device_indices.numel() != host_indices.numel():
        raise RuntimeError("FAST2D requires equal numbers of device and host slots")
    return list(
        zip(
            device_indices.cpu().long().flatten().tolist(),
            host_indices.cpu().long().flatten().tolist(),
        )
    )


def _check_slots(pairs, device_size, host_size):
    for d, h in pairs:
        if not (0 <= d < device_size and 0 <= h < host_size):
            raise RuntimeError("index exceeds component slot count")


def _check_layer(layer, host):
    if layer.device.type != "npu":
        raise RuntimeError("device state must be on NPU")
    if host.device.type != "cpu" or not host.is_pinned():
        raise RuntimeError("host state must use pinned memory")
    if (
        not host.is_contiguous()
        or host.dtype != layer.dtype
        or not host.numel()
        or not layer.numel()
    ):
        raise RuntimeError(
            "expected nonempty, matching state payloads and contiguous host storage"
        )
    if (
        layer.ndim < 2
        or host.ndim != layer.ndim + 2
        or host.size(2) != 1
        or host.shape[3:] != layer.shape[1:]
    ):
        raise RuntimeError("incompatible host and device slot shapes")
    slot = _payload(layer[0])
    if layer.stride(0) != slot.numel() or host.stride(1) != slot.numel():
        raise RuntimeError("slot payload must be contiguous")


def _payload_view(tensor, prefix_dims):
    """Flatten each dense payload in storage order, preserving outer pitches."""
    payload = _payload(tensor[(0,) * prefix_dims])
    return tensor.as_strided(
        (*tensor.shape[:prefix_dims], payload.numel()),
        (*tensor.stride()[:prefix_dims], 1),
    )


def _check_packed_buffers(device, host):
    if device.device.type != "npu":
        raise RuntimeError("device buffer must be on NPU")
    if host.device.type != "cpu" or not host.is_pinned():
        raise RuntimeError("host buffer must use pinned memory on CPU")
    if device.dtype != host.dtype:
        raise RuntimeError("device and host buffers must have the same dtype")
    if not host.is_contiguous():
        raise RuntimeError("host buffer must be contiguous")
    elements = _payload(device[0, 0]).numel()
    if device.size(1) != 1 and device.stride(1) != elements:
        raise RuntimeError("device slot pitch must be packed")
    if device.size(0) != 1 and device.stride(0) != device.size(1) * elements:
        raise RuntimeError("device layer pitch must be packed")


def _check_transfer_indices(device_indices, host_indices):
    if any(
        index.ndim != 1 or index.dtype != torch.long
        for index in (device_indices, host_indices)
    ):
        raise RuntimeError("indices must be one-dimensional int64 tensors")


def _packed_views(device, host):
    # The original Mamba/KV ACL kernels compute packed pitches from sizes,
    # rather than tensor strides. Keep that byte layout in the reference too.
    layers, slots = device.shape[:2]
    width = device[0, 0].numel()
    return (
        device.as_strided((layers, slots, width), (slots * width, width, 1)),
        host.as_strided(
            (host.size(0), layers, 1, width), (layers * width, width, width, 1)
        ),
    )


def _write_slots(dst, indices, values):
    indices = indices.to(device="cpu", dtype=torch.long).reshape(-1)
    # ACL submits copies in input order. index_copy_ with repeated destinations
    # does not guarantee that order on accelerators, so retain ordered writes.
    if indices.unique().numel() != indices.numel():
        for i, index in enumerate(indices.tolist()):
            dst[index].copy_(values[i])
    else:
        dst.index_copy_(0, indices.to(dst.device), values)


def _copy_per_layer(host, device, host_indices, device_indices, layer_id):
    # SGLang MambaPoolHost._copy_tensor_pf_lf: sync Torch fallback.
    host_indices = host_indices.to(dtype=torch.long, device=host.device).reshape(-1)
    values = host.select(1, layer_id).index_select(0, host_indices).select(1, 0)
    _write_slots(device, device_indices, values.to(device=device.device))


def _copy_all_layers(device, host, device_indices, host_indices):
    # SGLang MambaPoolHost._copy_tensor_all_layers_lf_pf: per-layer fallback.
    device_indices = device_indices.to(dtype=torch.long, device=device.device).reshape(
        -1
    )
    values = (
        device.index_select(1, device_indices)
        .movedim(0, 1)
        .unsqueeze(2)
        .contiguous()
        .to(device=host.device)
    )
    _write_slots(host, host_indices, values)


def _copy_state(device, host, device_indices, host_indices, direction):
    if not device_indices.numel():
        return
    if direction.value == _D2H:
        _copy_all_layers(device, host, device_indices, host_indices)
    else:
        for layer_id in range(device.size(0)):
            _copy_per_layer(
                host, device[layer_id], host_indices, device_indices, layer_id
            )


def _copy_packed_state(device, host, device_indices, host_indices, direction):
    _check_packed_buffers(device, host)
    if not device_indices.numel():
        return
    # Keep SGLang's gather/scatter on CPU for H2D. Snapshot every slot so that
    # unselected pages survive, and preserve dense payload permutations for ACL's
    # physical storage order. Copy back through the original shape, not a flat
    # NPU view: odd-width int8 payloads must not overwrite neighbouring pages.
    target = device
    if direction.value != _D2H:
        target = device.to(device="cpu", copy=True, memory_format=torch.preserve_format)
    device_view, host_view = _packed_views(target, host)
    _copy_state(device_view, host_view, device_indices, host_indices, direction)
    if direction.value != _D2H:
        device.copy_(target)


def transfer_kv_per_layer_direct_pf_lf(
    src, dst, src_indices, dst_indices, layer_id, flags
):
    pairs = _slot_pairs(dst_indices, src_indices, flags)
    if not pairs:
        return
    _check_layer(dst, src)
    if not 0 <= layer_id < src.size(1):
        raise RuntimeError("layer_id is outside host state layer count")
    _check_slots(pairs, dst.size(0), src.size(0))
    target = dst.to(device="cpu", copy=True, memory_format=torch.preserve_format)
    device = _payload_view(target, 1)
    host = src.view(src.size(0), src.size(1), 1, device.size(-1))
    _copy_per_layer(host, device, src_indices, dst_indices, layer_id)
    dst.copy_(target)


def transfer_kv_all_layer_direct_lf_pf(
    device_states, host_states, device_indices, host_indices, flags
):
    if not device_states or len(device_states) != len(host_states):
        raise RuntimeError("expected matching nonempty component lists")
    pairs = _slot_pairs(device_indices, host_indices, flags)
    if not pairs:
        return
    layers = device_states[0].size(0)
    for dev, host in zip(device_states, host_states):
        if (
            not dev.numel()
            or dev.ndim < 3
            or host.ndim < 2
            or dev.size(0) != host.size(1)
            or dev.size(0) < layers
        ):
            raise RuntimeError("incompatible component layer counts or shape")
        _check_layer(dev[0], host)
        if dev.stride(0) < dev.size(1) * dev.stride(1):
            raise RuntimeError("device layer pitch overlaps adjacent slots")
        _check_slots(pairs, dev.size(1), host.size(0))
    for dev, host in zip(device_states, host_states):
        device = _payload_view(dev[:layers], 2)
        host = host.view(host.size(0), host.size(1), 1, device.size(-1))
        _copy_all_layers(device, host[:, :layers], device_indices, host_indices)


def transfer_mamba_state(device_buf, host_buf, device_indices, host_indices, direction):
    if (
        device_buf.ndim < 2
        or host_buf.ndim < 2
        or not device_buf.numel()
        or not host_buf.numel()
        or device_buf.size(0) != host_buf.size(1)
        or device_buf[0, 0].numel() != host_buf[0, 0].numel()
    ):
        raise RuntimeError("incompatible Mamba state buffers")
    _check_transfer_indices(device_indices, host_indices)
    if device_indices.numel() != host_indices.numel() or direction.value not in (1, 2):
        raise RuntimeError("invalid Mamba transfer indices or direction")
    for d, h in zip(device_indices.tolist(), host_indices.tolist()):
        d, h = int(d), int(h)
        if not (0 <= d < device_buf.size(1) and 0 <= h < host_buf.size(0)):
            raise RuntimeError("slot index out of range")
    _copy_packed_state(device_buf, host_buf, device_indices, host_indices, direction)


def transfer_kv_dim_exchange(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k,
    host_index_k,
    device_index_k_scale,
    host_index_k_scale,
    page_size,
    direction,
    flags,
):
    _check_transfer_indices(device_indices, host_indices)
    if (
        page_size <= 0
        or device_indices.numel() != host_indices.numel()
        or device_indices.numel() % page_size
        or direction.value not in (1, 2)
        or flags.value & 2 != 2
    ):
        raise RuntimeError("invalid KV transfer parameters")
    if not device_k.numel() or not host_k.numel():
        raise RuntimeError("K caches must not be empty")
    if device_index_k_scale is not None and device_index_k_scale.ndim == 4:
        device_index_k_scale = device_index_k_scale.unsqueeze(-1)
    pairs = [
        (device_k, host_k),
    ]
    if device_v.numel() and host_v.numel():
        if (
            device_v.ndim != 5
            or host_v.ndim != 5
            or device_v.shape[:4] != device_k.shape[:4]
            or host_v.shape[:4] != host_k.shape[:4]
        ):
            raise RuntimeError("K/V layer, page, page-size and head counts must match")
        if device_v.dtype != device_k.dtype:
            raise RuntimeError("K/V caches must have the same dtype")
        if device_v.device != device_k.device:
            raise RuntimeError("K/V caches must be on the same NPU")
        if device_v.size(4) != host_v.size(4):
            raise RuntimeError("device and host V head dimensions must match")
        _check_packed_buffers(device_v, host_v)
        pairs.append((device_v, host_v))
    for dev, host in (
        (device_index_k, host_index_k),
        (device_index_k_scale, host_index_k_scale),
    ):
        if dev is not None and host is not None:
            pairs.append((dev, host))
    if any(
        (index[::page_size] < 0).any().item()
        for index in (device_indices, host_indices)
    ):
        raise RuntimeError("transfer indices must be non-negative")
    # ACL uses only the first token index in each group to identify a whole page.
    device_pages = device_indices[::page_size].to(dtype=torch.long) // page_size
    host_pages = host_indices[::page_size].to(dtype=torch.long) // page_size
    for dev, host in pairs:
        if (
            not dev.numel()
            or not host.numel()
            or dev.ndim != 5
            or host.ndim != 5
            or dev.size(0) != host.size(1)
            or dev.size(2) != page_size
            or host.size(2) != page_size
            or dev.shape[2:] != host.shape[2:]
        ):
            raise RuntimeError("incompatible device and host KV page layouts")
        _check_slots(
            zip(device_pages.cpu().tolist(), host_pages.cpu().tolist()),
            dev.size(1),
            host.size(0),
        )
        _copy_packed_state(dev, host, device_pages, host_pages, direction)
