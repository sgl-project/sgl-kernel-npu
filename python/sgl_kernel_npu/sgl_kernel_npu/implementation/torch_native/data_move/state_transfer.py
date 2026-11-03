"""PyTorch implementations of layer-first / page-first transfers."""

from ....operators.data_move._types import TransferDirection


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
    for d, h in pairs:
        _payload(dst[d]).copy_(_payload(src[h, layer_id, 0]), non_blocking=True)


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
        for d, h in pairs:
            for layer in range(layers):
                _payload(host[h, layer, 0]).copy_(
                    _payload(dev[layer, d]), non_blocking=True
                )


def _copy(dev, host, direction):
    src, dst = (
        (host, dev) if direction.value == TransferDirection.H2D.value else (dev, host)
    )
    dst.copy_(src, non_blocking=True)


def transfer_mamba_state(device_buf, host_buf, device_indices, host_indices, direction):
    if (
        device_buf.ndim < 2
        or host_buf.ndim < 2
        or not device_buf.numel()
        or not host_buf.numel()
        or device_buf.size(0) != host_buf.size(1)
        or device_buf.dtype != host_buf.dtype
        or device_buf[0, 0].numel() != host_buf[0, 0].numel()
    ):
        raise RuntimeError("incompatible Mamba state buffers")
    if device_indices.numel() != host_indices.numel() or direction.value not in (1, 2):
        raise RuntimeError("invalid Mamba transfer indices or direction")
    for d, h in zip(device_indices.tolist(), host_indices.tolist()):
        d, h = int(d), int(h)
        if not (0 <= d < device_buf.size(1) and 0 <= h < host_buf.size(0)):
            raise RuntimeError("slot index out of range")
        _copy(device_buf[:, d], host_buf[h].reshape_as(device_buf[:, d]), direction)


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
        (device_v, host_v),
        (device_index_k, host_index_k),
        (device_index_k_scale, host_index_k_scale),
    ]
    for dev, host in pairs:
        if dev is None or host is None or not dev.numel() or not host.numel():
            continue
        if (
            dev.ndim != 5
            or host.ndim != 5
            or dev.size(0) != host.size(1)
            or dev.size(2) != page_size
            or host.size(2) != page_size
        ):
            raise RuntimeError("incompatible device and host KV page layouts")
        for d, h in zip(
            device_indices[::page_size].tolist(), host_indices[::page_size].tolist()
        ):
            block = dev[:, d // page_size]
            _copy(block, host[h // page_size].reshape_as(block), direction)
