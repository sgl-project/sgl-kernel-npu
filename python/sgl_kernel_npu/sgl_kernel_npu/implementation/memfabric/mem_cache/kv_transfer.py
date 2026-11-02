"""MemFabric KV transfers on the caller's current NPU stream.

The caller owns initialized MemFabric resources and registered Host storage.
Views, index uploads and metadata stay alive until their completion event.
"""

from collections import deque
from threading import Lock

import torch
import triton
import triton.language as tl

_LENGTH_DTYPE = torch.int32
# Current Memfabric LOCAL_URMA capacity: HCOMM_BATCH_TRANSFER_MAX_DESC_NUM
# in src/hybm/csrc/under_api/dl_hcomm_api.h; see offload_sparse_copy's contract.
MAX_SPARSE_COPY_ENTRIES = 8190
_submit_lock = Lock()
_last_sparse_event = {}
_inflight = deque()


def _check_component(device, host, page_size, index_device):
    if device.device != index_device or host.device.type != "cpu":
        raise ValueError("KV components must be on the index NPU and CPU respectively")
    if device.ndim != 5 or host.ndim != 5:
        raise ValueError(
            "expected Device LF and normalized Host PF five-dimensional buffers"
        )
    if (
        device.dtype != host.dtype
        or device.shape[0] != host.shape[1]
        or device.shape[0] <= 0
    ):
        raise ValueError("KV component dtypes and positive layer counts must match")
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


@triton.jit(do_not_specialize=["device_info", "host_info", "pages"])
def _build_kv_sparse_metadata_kernel(
    device_indices,
    host_indices,
    src,
    dst,
    lengths,
    count,
    device_info,
    host_info,
    pages,
    LAYOUTS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    H2D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    entry = offsets.to(tl.int64)
    component = tl.program_id(1)
    layer_offset = 0
    for c in tl.static_range(len(LAYOUTS)):
        (
            layers,
            device_layer_stride,
            device_page_stride,
            host_layer_stride,
            host_page_stride,
            page_bytes,
        ) = LAYOUTS[c]
        if component == c:
            device_base, device_pages = device_info[c]
            host_base, host_pages = host_info[c]
            mask = offsets < pages * layers
            output = pages * layer_offset + entry
            page, layer = entry // layers, entry % layers
            dt = tl.load(device_indices + page * PAGE_SIZE, mask=mask, other=0)
            ht = tl.load(host_indices + page * PAGE_SIZE, mask=mask, other=0)
            dp, hp = dt // PAGE_SIZE, ht // PAGE_SIZE
            valid = (
                (dt >= 0)
                & (ht >= 0)
                & (dt % PAGE_SIZE == 0)
                & (ht % PAGE_SIZE == 0)
                & (dp < device_pages)
                & (hp < host_pages)
            )
            daddr = device_base.to(tl.uint64) + (
                layer * device_layer_stride + dp * device_page_stride
            ).to(tl.uint64)
            haddr = host_base.to(tl.uint64) + (
                layer * host_layer_stride + hp * host_page_stride
            ).to(tl.uint64)
            # Invalid descriptors keep safe bases and are rejected by LOCAL_URMA.
            daddr = tl.where(valid, daddr, device_base.to(tl.uint64))
            haddr = tl.where(valid, haddr, host_base.to(tl.uint64))
            tl.store(src + output, haddr if H2D else daddr, mask=mask)
            tl.store(dst + output, daddr if H2D else haddr, mask=mask)
            tl.store(lengths + output, tl.where(valid, page_bytes, -1), mask=mask)
        layer_offset += layers
    if (component == 0) & (tl.program_id(0) == 0):
        tl.store(count, pages * layer_offset)


def build_kv_sparse_metadata(
    components, device_indices, host_indices, page_size, direction
):
    """Build all address tables in one launch; transfer_kv validates the inputs."""
    pages = device_indices.numel() // page_size
    total = pages * sum(dev.shape[0] for dev, _ in components)
    if total > MAX_SPARSE_COPY_ENTRIES:
        raise ValueError(
            f"single-batch sparse_copy supports at most {MAX_SPARSE_COPY_ENTRIES} "
            f"descriptors, got {total}"
        )
    device = device_indices.device
    src = torch.empty((total,), dtype=torch.int64, device=device)
    dst = torch.empty_like(src)
    lengths = torch.empty((total,), dtype=_LENGTH_DTYPE, device=device)
    count = torch.empty((1,), dtype=torch.int32, device=device)
    layouts = []
    for dev, host in components:
        item_size = dev.element_size()
        layouts.append(
            (
                dev.shape[0],
                dev.stride(0) * item_size,
                dev.stride(1) * item_size,
                host.stride(1) * item_size,
                host.stride(0) * item_size,
                page_size * dev.shape[3] * dev.shape[4] * item_size,
            )
        )
    max_entries = pages * max(dev.shape[0] for dev, _ in components)
    _build_kv_sparse_metadata_kernel[(triton.cdiv(max_entries, 128), len(components))](
        device_indices,
        host_indices,
        src,
        dst,
        lengths,
        count,
        tuple((dev.data_ptr(), dev.shape[1]) for dev, _ in components),
        tuple((host.data_ptr(), host.shape[0]) for _, host in components),
        pages,
        LAYOUTS=tuple(layouts),
        PAGE_SIZE=page_size,
        H2D=direction == 1,
        BLOCK=128,
    )
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
    return (
        dev[device_start : device_start + count],
        host[:, host_start : host_start + count],
    )


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


def _device_indices(indices, device, keepalive):
    if indices.device.type == "cpu":
        pinned = indices.pin_memory()
        keepalive.append(pinned)
        indices = pinned.to(device, non_blocking=True)
    elif indices.device != device:
        raise ValueError("KV indices must be on CPU or the KV buffer's NPU")
    return indices.contiguous()


def _submit_aicpu(offload, components, di, hi, page_size, direction, device, keepalive):
    scene = offload._get_scene()
    if scene not in (offload.Scene.LOCAL, offload.Scene.LOCAL_URMA):
        raise RuntimeError(
            "sparse_copy requires initialized MemFabric LOCAL or LOCAL_URMA"
        )
    # Preserve sparse_copy's existing address contract: pass the Host pool
    # pointers through unchanged, as in transfer_kv_sparse_copy.
    src, dst, lengths, count = build_kv_sparse_metadata(
        components, di, hi, page_size, direction
    )
    keepalive.extend((src, dst, lengths, count))
    copy_direction = (
        offload.CopyDirection.H2D if direction == 1 else offload.CopyDirection.D2H
    )
    ret = offload.sparse_copy(src, dst, lengths, count, device, copy_direction)
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
    for dev, host in components:
        _check_component(dev, host, page_size, device)
    if not device_indices.numel():
        return
    from memfabric_hybrid import offload

    with _submit_lock, torch.npu.device(device):
        while _inflight and _inflight[0][0].query():
            _inflight.popleft()
        stream = torch.npu.current_stream()
        if backend == "aicpu" and device in _last_sparse_event:
            stream.wait_event(_last_sparse_event[device])
        keepalive = [tensor for pair in components for tensor in pair]
        keepalive.extend((device_indices, host_indices))
        try:
            di = _device_indices(device_indices, device, keepalive)
            hi = _device_indices(host_indices, device, keepalive)
            keepalive.extend((di, hi))
            # Strong references survive until done.query() confirms completion,
            # including tensors allocated on a different stream.
            if backend == "aicpu":
                _submit_aicpu(
                    offload, components, di, hi, page_size, direction, device, keepalive
                )
            else:
                _submit_aiv(
                    offload,
                    components,
                    di,
                    hi,
                    page_size,
                    direction,
                    device,
                    host_layout,
                    keepalive,
                )
        finally:
            # Submitted metadata and pinned uploads must survive failures too.
            done = torch.npu.Event()
            done.record(stream)
            if backend == "aicpu":
                _last_sparse_event[device] = done
            _inflight.append((done, keepalive))
