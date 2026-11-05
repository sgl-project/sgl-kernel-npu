"""MemFabric KV transfers on the caller's current NPU stream.

The caller owns initialized MemFabric resources and Host storage.
Eager inputs, sparse-copy metadata and AIV pinned staging stay alive until their
completion event. Captured metadata belongs to the graph memory pool; callers
retain graph inputs and MF resources.
"""

from collections import deque
from threading import Lock

import torch

_LENGTH_DTYPE = torch.int32
# AICPU input limit: HYBM_SPARSE_COPY_MAX_RECORDS / HYBM_SDMA_MAX_RECORDS.
MAX_SPARSE_COPY_ENTRIES = 65536
_submit_lock = Lock()
_last_sparse_submission = {}
_inflight = deque()


def _check_component(device, host, page_size, index_device):
    if device.device != index_device or host.device.type != "cpu":
        raise ValueError("KV components must be on the index NPU and CPU respectively")
    # _layer_views already checked dimensions and selected equal layer counts.
    if device.dtype != host.dtype or device.shape[0] <= 0:
        raise ValueError(
            "KV component dtypes must match and layer count must be positive"
        )
    if device.shape[2:] != host.shape[2:] or device.shape[2] != page_size:
        raise ValueError("KV component page shapes must match")
    for tensor in (device, host):
        if (
            tensor.stride(4) != 1
            or tensor.stride(3) != tensor.shape[4]
            or tensor.stride(2) != tensor.shape[3] * tensor.shape[4]
        ):
            raise ValueError("each (page,layer) payload must be contiguous")
    page_bytes = page_size * device.shape[3] * device.shape[4] * device.element_size()
    if not 0 < page_bytes <= torch.iinfo(_LENGTH_DTYPE).max:
        raise ValueError(
            f"sparse_copy page byte count {page_bytes} must fit a positive {_LENGTH_DTYPE}"
        )


def build_kv_sparse_metadata(
    components, device_indices, host_indices, page_size, direction
):
    """Build sparse-copy inputs with PyTorch operations on the indices' NPU."""
    pages = device_indices.numel() // page_size
    total = pages * sum(dev.shape[0] for dev, _ in components)
    if total > MAX_SPARSE_COPY_ENTRIES:
        raise ValueError(
            f"single-batch sparse_copy supports at most {MAX_SPARSE_COPY_ENTRIES} "
            f"descriptors, got {total}"
        )
    device = device_indices.device
    dt, ht = device_indices[::page_size], host_indices[::page_size]
    dp, hp = dt // page_size, ht // page_size
    aligned = (dt >= 0) & (ht >= 0) & (dt % page_size == 0) & (ht % page_size == 0)
    src_parts, dst_parts, length_parts = [], [], []
    for dev, host in components:
        layers = dev.shape[0]
        layer = torch.arange(layers, dtype=torch.int64, device=device)
        item_size = dev.element_size()
        valid = aligned & (dp < dev.shape[1]) & (hp < host.shape[0])
        daddr = (
            dev.data_ptr()
            + dp[:, None] * (dev.stride(1) * item_size)
            + layer[None, :] * (dev.stride(0) * item_size)
        )
        haddr = (
            host.data_ptr()
            + hp[:, None] * (host.stride(0) * item_size)
            + layer[None, :] * (host.stride(1) * item_size)
        )
        # Preserve rejected descriptors: safe pool bases and a negative length.
        daddr = torch.where(valid[:, None], daddr, dev.data_ptr()).reshape(-1)
        haddr = torch.where(valid[:, None], haddr, host.data_ptr()).reshape(-1)
        src_parts.append(haddr if direction == 1 else daddr)
        dst_parts.append(daddr if direction == 1 else haddr)
        page_bytes = page_size * dev.shape[3] * dev.shape[4] * item_size
        length_parts.append(
            torch.where(valid, page_bytes, -1)
            .to(_LENGTH_DTYPE)
            .repeat_interleave(layers)
        )
    src = torch.cat(src_parts)
    dst = torch.cat(dst_parts)
    lengths = torch.cat(length_parts)
    count = torch.full((1,), total, dtype=torch.int32, device=device)
    return src, dst, lengths, count


def _layer_views(dev, host, host_layout, layers):
    if dev.ndim != 5 or host.ndim != 5:
        raise ValueError(
            "KV components must be five-dimensional (Device scales may be four-dimensional)"
        )
    if host_layout == "layer_first":
        host = host.transpose(0, 1)
    device_start, host_start, count = layers
    count = dev.shape[0] - device_start if count is None else count
    if (
        device_start < 0
        or host_start < 0
        or count < 0
        or device_start + count > dev.shape[0]
        or host_start + count > host.shape[1]
    ):
        raise ValueError("KV layer selection exceeds Device or Host storage")
    if device_start != 0 or count != dev.shape[0]:
        dev = dev.narrow(0, device_start, count)
    if host_start != 0 or count != host.shape[1]:
        host = host.narrow(1, host_start, count)
    return dev, host


def select_components(
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k,
    host_index_k,
    device_index_k_scale,
    host_index_k_scale,
    host_layout,
    kv_layers,
    index_layers,
):
    """Select metadata views; optional components retain legacy paired-presence semantics."""
    if host_layout not in ("page_first", "layer_first"):
        raise ValueError("host_layout must be page_first or layer_first")
    if device_k is None or host_k is None or not device_k.numel() or not host_k.numel():
        raise ValueError("K buffers must be nonempty")
    if device_v is None or host_v is None:
        raise ValueError(
            "V buffers must be tensors; use an empty Device V for no V payload"
        )
    components = [_layer_views(device_k, host_k, host_layout, kv_layers)]
    if device_v.numel():
        if not host_v.numel():
            raise ValueError("nonempty Device V requires a nonempty Host V")
        components.append(_layer_views(device_v, host_v, host_layout, kv_layers))
    if index_layers[2] != 0:
        for dev, host, is_scale in (
            (device_index_k, host_index_k, False),
            (device_index_k_scale, host_index_k_scale, True),
        ):
            if dev is not None and host is not None:
                if dev.numel() or host.numel():
                    if not dev.numel() or not host.numel():
                        raise ValueError(
                            "paired Indexer components must both be nonempty"
                        )
                    if is_scale and dev.ndim == 4:
                        dev = dev.unsqueeze(-1)
                    components.append(
                        _layer_views(dev, host, host_layout, index_layers)
                    )
    return components


def _submit_aicpu(offload, components, di, hi, page_size, direction, device, keepalive):
    # Preserve sparse_copy's existing address contract: pass the Host pool
    # pointers through unchanged, as in transfer_kv_sparse_copy.
    src, dst, lengths, count = build_kv_sparse_metadata(
        components, di, hi, page_size, direction
    )
    keepalive.extend((src, dst, lengths, count))
    ret = offload.sparse_copy(src, dst, lengths, count, device)
    if ret != 0:
        raise RuntimeError(f"MemFabric sparse_copy failed: code={ret}")


def _resolve_host_dva(host):
    # Mirrors the mf_aiv host_memory contract, device-validated there: the
    # registration table (host_register + host_get_device_pointer at
    # allocation time, interior views by byte offset) is the ONLY DVA source;
    # unregistered MF memory is an error with no HVA/get_dva fallback.
    from sgl_kernel_npu.host_memory import resolve_host_dva

    return resolve_host_dva(host)


def _submit_aiv(
    offload, components, di, hi, page_size, direction, device, host_layout, keepalive
):
    # Host tensors have PF-shaped metadata views even when their storage is LF.
    # Actual page/layer strides preserve the original physical layout.
    # The C++ kv_exchange adapter builds the device-resident meta (the same
    # component/pitch/width/layer contract as the aicpu build_kv_sparse_
    # metadata path) and launches the AIV kernel; it returns the pinned
    # staging tensor that carried the meta upload, which stays alive until
    # the completion event retires the keepalive list.
    host_dvas = [_resolve_host_dva(host) for _, host in components]
    staging = torch.ops.npu.kv_exchange(
        di,
        hi,
        [dev for dev, _ in components],
        [host for _, host in components],
        host_dvas,
        page_size,
        direction,
        int(host_layout == "layer_first"),
    )
    if staging is not None:
        keepalive.append(staging)


def transfer_kv(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layout,
    direction,
    kv_layers,
    index_layers,
    backend,
):
    """Submit one selected pool/range without synchronizing the caller's stream."""
    if page_size <= 0 or device_indices.ndim != 1 or host_indices.ndim != 1:
        raise ValueError("expected positive page_size and 1-D token indices")
    if device_indices.dtype != torch.int64 or host_indices.dtype != torch.int64:
        raise ValueError("KV indices must have dtype int64")
    if (
        device_indices.numel() != host_indices.numel()
        or device_indices.numel() % page_size
    ):
        raise ValueError("indices must describe equal numbers of complete pages")
    if direction not in (1, 2) or backend not in ("aicpu", "aiv"):
        raise ValueError("expected H2D (1) or D2H (2), and aicpu or aiv")
    components = select_components(
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        host_layout,
        kv_layers,
        index_layers,
    )
    device = device_k.device
    if device.type != "npu":
        raise ValueError("KV Device buffers must be NPU tensors")
    if device_indices.device != device or host_indices.device != device:
        raise ValueError("KV indices must be on the KV buffer's NPU")
    for dev, host in components:
        _check_component(dev, host, page_size, device)
    if not device_indices.numel():
        return
    # Only the AICPU op goes through the MemFabric offload entity; the AIV
    # path resolves host pointers through the registered-host DVA table
    # (MemFabric or aclrt registration), so acl-only processes never touch
    # memfabric_hybrid below.
    offload = None
    if backend == "aicpu":
        from memfabric_hybrid import offload

    with _submit_lock, torch.npu.device(device):
        if torch.npu.is_current_stream_capturing():
            if backend != "aicpu":
                raise RuntimeError("NPU graph capture currently requires impl='aicpu'")
            # Allocations made here belong to the graph pool. Eager completion
            # events cannot govern replay lifetimes, and event queries are not
            # capture-safe. The caller orders replays and retains static inputs.
            _submit_aicpu(offload, components, device_indices.contiguous(), host_indices.contiguous(), page_size, direction, device, [])  # fmt: skip
            return
        while _inflight and _inflight[0][0].query():
            _inflight.popleft()
        stream = torch.npu.current_stream()
        if backend == "aicpu":
            previous = _last_sparse_submission.get(device)
            if previous is not None:
                previous_stream, previous_done = previous
                # Same-stream submissions are already ordered through completion.
                if stream != previous_stream:
                    stream.wait_event(previous_done)
        keepalive = [tensor for pair in components for tensor in pair]
        keepalive.extend((device_indices, host_indices))
        try:
            di = device_indices.contiguous()
            hi = host_indices.contiguous()
            # New contiguous copies use this stream's allocator. Keep only
            # the original inputs, which may have been allocated on other streams.
            if backend == "aicpu":
                _submit_aicpu(offload, components, di, hi, page_size, direction, device, keepalive)  # fmt: skip
            else:
                _submit_aiv(offload, components, di, hi, page_size, direction, device, host_layout, keepalive)  # fmt: skip
        finally:
            # Inputs, sparse-copy metadata and AIV staging survive failures too.
            done = torch.npu.Event()
            done.record(stream)
            if backend == "aicpu":
                _last_sparse_submission[device] = (stream, done)
            _inflight.append((done, keepalive))
