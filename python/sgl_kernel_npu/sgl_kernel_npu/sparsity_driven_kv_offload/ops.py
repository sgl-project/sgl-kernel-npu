import ctypes
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch


_SUPPORTED_LRU_CACHE_CAPACITIES = (2048, 4096, 6144, 8192)


def _ctype_for_dtype(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return ctypes.c_uint16
    if dtype == torch.float32:
        return ctypes.c_float
    if dtype == torch.float64:
        return ctypes.c_double
    if dtype == torch.int8:
        return ctypes.c_int8
    if dtype == torch.uint8:
        return ctypes.c_uint8
    if dtype == torch.int16:
        return ctypes.c_int16
    if dtype == torch.int32:
        return ctypes.c_int32
    if dtype == torch.int64:
        return ctypes.c_int64
    if dtype == torch.bool:
        return ctypes.c_bool
    raise TypeError(f"unsupported shm tensor dtype: {dtype}")


def create_shm_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int = 0,
    name: str = "",
) -> Tuple[torch.Tensor, int, int]:
    """Create host shared memory and register it to an NPU device.

    Returns ``(host_tensor, host_ptr, dev_ptr)``. ``host_tensor`` is a CPU
    tensor backed by the registered shared memory. ``dev_ptr`` is the
    device-visible address and can be passed to sparse KV kernels through
    ``src_ptr``/``dst_ptr``.
    """
    shape_tuple = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape_tuple):
        raise ValueError(f"shape dimensions must be non-negative, got {shape_tuple}")

    numel = int(prod(shape_tuple))
    elem_size = torch.empty((), dtype=dtype).element_size()
    size = numel * elem_size
    if size <= 0:
        raise ValueError(f"shm tensor size must be positive, got shape={shape_tuple}")

    host_ptr, dev_ptr = torch.ops.npu.shm_allocator_create_and_register(
        size, device_id, name
    )
    buffer_type = _ctype_for_dtype(dtype) * numel
    buffer = buffer_type.from_address(host_ptr)
    tensor = torch.frombuffer(buffer, dtype=dtype).view(shape_tuple)
    if tensor.element_size() != elem_size:
        raise RuntimeError("shm tensor element size mismatch")
    tensor.zero_()
    return tensor, int(host_ptr), int(dev_ptr)


def free_shm(device_id: int = 0) -> None:
    """Free all shared-memory allocations registered by this process."""
    torch.ops.npu.shm_allocator_free_all(device_id)


def _infer_rows_and_block_bytes(
    tensor: torch.Tensor, address_ndims: int, name: str
) -> Tuple[int, int]:
    if address_ndims <= 0 or address_ndims >= tensor.dim():
        raise ValueError(
            f"{name}_address_ndims must be in [1, {tensor.dim() - 1}], "
            f"got {address_ndims}"
        )

    rows = prod(tensor.shape[:address_ndims])
    block_elements = prod(tensor.shape[address_ndims:])
    return int(rows), int(block_elements * tensor.element_size())


def unidex_copy_inplace(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
    dst_ptr: Optional[int] = None,
) -> torch.Tensor:
    """Copy selected logical rows from ``src`` into ``dst`` in place.

    ``src_ptr`` and ``dst_ptr`` may override the Tensor addresses with
    device-visible shared-memory addresses. The caller owns those allocations
    and must keep them alive until work on the current NPU stream completes.
    """
    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, dst_block_bytes = _infer_rows_and_block_bytes(
        dst, dst_address_ndims, "dst"
    )
    if src_block_bytes != dst_block_bytes:
        raise ValueError(
            "src and dst logical rows must have the same byte size, got "
            f"{src_block_bytes} and {dst_block_bytes}"
        )
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            f"src and dst must have the same dtype, got {src.dtype} and {dst.dtype}"
        )

    torch.ops.npu.unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        src_block_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
        dst_ptr,
    )
    return dst


def slot_map_lookup(
    slot_map: torch.Tensor,
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    block_dim: int = 0,
    pos_mask_size: Optional[int] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Return cache-hit flags and slot positions for ``topk_indices``.

    When ``pos_mask_size`` is provided, also return an int32 mask with shape
    ``[bs, pos_mask_size]``. For every cache hit at position ``pos``, the
    kernel sets ``position_mask[b, pos] = 1``. Hit positions outside the mask
    range are ignored. Omitting ``pos_mask_size`` preserves the legacy
    two-output API and disables position-mask writes in the kernel.
    """
    token_on_device = torch.empty_like(topk_indices, dtype=torch.int32)
    device_token_pos = torch.empty_like(topk_indices, dtype=torch.int32)
    if pos_mask_size is None:
        effective_pos_mask_size = 0
        position_mask = torch.empty(
            (topk_indices.size(0), 0),
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        effective_pos_mask_size = int(pos_mask_size)
        if effective_pos_mask_size <= 0:
            raise ValueError(
                f"pos_mask_size must be positive when provided, got {pos_mask_size}"
            )
        if effective_pos_mask_size % 8 != 0:
            raise ValueError(
                "pos_mask_size must be a multiple of 8 for aligned atomic mask "
                f"updates, got {pos_mask_size}"
            )
        position_mask = torch.empty(
            (topk_indices.size(0), effective_pos_mask_size),
            dtype=torch.int32,
            device=topk_indices.device,
        )
    torch.ops.npu.slot_map_lookup(
        slot_map,
        req_indices,
        topk_indices,
        token_on_device,
        device_token_pos,
        position_mask,
        effective_pos_mask_size,
        block_dim,
    )
    if pos_mask_size is not None:
        return token_on_device, device_token_pos, position_mask
    return token_on_device, device_token_pos


def _validate_fused_timestamp_lru_inputs(
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    device_token_pos: torch.Tensor,
    hit_position_mask: torch.Tensor,
    device_lru_slots: torch.Tensor,
    device_lru_slot_stamps: torch.Tensor,
) -> None:
    if req_indices.dtype != torch.int32:
        raise ValueError(f"req_indices must be int32, got {req_indices.dtype}")
    if topk_indices.dtype != torch.int32:
        raise ValueError(f"topk_indices must be int32, got {topk_indices.dtype}")
    if device_token_pos.dtype != torch.int32:
        raise ValueError(
            f"device_token_pos must be int32, got {device_token_pos.dtype}"
        )
    if hit_position_mask.dtype != torch.int32:
        raise ValueError(
            f"hit_position_mask must be int32, got {hit_position_mask.dtype}"
        )
    if topk_indices.dim() != 2 or topk_indices.size(1) != 2048:
        raise ValueError(
            "fused timestamp LRU requires topk_indices shape [batch, 2048], "
            f"got {tuple(topk_indices.shape)}"
        )
    if (
        device_lru_slots.dim() != 2
        or device_lru_slots.size(1) not in _SUPPORTED_LRU_CACHE_CAPACITIES
    ):
        raise ValueError(
            "fused timestamp LRU requires device_lru_slots shape "
            "[request_rows, cache_capacity] with cache_capacity in "
            f"{_SUPPORTED_LRU_CACHE_CAPACITIES}, got "
            f"{tuple(device_lru_slots.shape)}"
        )
    if device_lru_slot_stamps.shape != device_lru_slots.shape:
        raise ValueError(
            "device_lru_slot_stamps shape must match device_lru_slots, got "
            f"{tuple(device_lru_slot_stamps.shape)} and "
            f"{tuple(device_lru_slots.shape)}"
        )
    cache_capacity = device_lru_slots.size(1)
    if hit_position_mask.shape != (topk_indices.size(0), cache_capacity):
        raise ValueError(
            "fused timestamp LRU requires hit_position_mask shape "
            "[batch, cache_capacity], got "
            f"{tuple(hit_position_mask.shape)} with "
            f"cache_capacity={cache_capacity}"
        )


def fused_timestamp_lru_metadata_update(
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    device_token_pos: torch.Tensor,
    hit_position_mask: torch.Tensor,
    device_lru_slots: torch.Tensor,
    device_lru_slot_stamps: torch.Tensor,
    max_context_len: int,
    stamp_max: int = (1 << 24) - 1,
    block_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select timestamp-LRU victims and update the ordered LRU state.

    The operator is specialized for ``topk=2048`` and supports cache capacities
    ``2048``, ``4096``, ``6144``, and ``8192``. ``hit_position_mask`` is the
    capacity-sized mask returned by ``slot_map_lookup``.
    ``device_lru_slots`` and ``device_lru_slot_stamps`` are aligned pairs in
    descending timestamp order and are updated in place. The returned tuple is
    ``(victim_slots, miss_counts)``. ``victim_slots`` contains one physical
    victim per miss, or ``-1`` for hit/invalid top-k positions of valid
    requests. Output rows for invalid request IDs are undefined and must be
    ignored by the caller's valid mask.

    Request IDs start at row 0. Valid request rows in one launch must be unique
    because one AIV owns each row.
    """
    _validate_fused_timestamp_lru_inputs(
        req_indices,
        topk_indices,
        device_token_pos,
        hit_position_mask,
        device_lru_slots,
        device_lru_slot_stamps,
    )
    return torch.ops.npu.fused_timestamp_lru_metadata_update(
        req_indices,
        topk_indices,
        device_token_pos,
        hit_position_mask,
        device_lru_slots,
        device_lru_slot_stamps,
        int(max_context_len),
        int(stamp_max),
        int(block_dim),
    )


def fused_timestamp_lru_metadata_update_with_probation(
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    device_token_pos: torch.Tensor,
    hit_position_mask: torch.Tensor,
    device_lru_slots: torch.Tensor,
    device_lru_slot_stamps: torch.Tensor,
    max_context_len: int,
    probation_age: int,
    stamp_max: int = (1 << 24) - 1,
    block_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update timestamp-LRU metadata with probationary miss insertion.

    Hits are reset to age zero. Each newly filled miss slot starts at
    ``probation_age`` and is stably inserted into the descending-age LRU order,
    so a one-time miss does not immediately receive the same MRU status as a
    hit. Passing ``probation_age=0`` is behaviorally compatible with
    :func:`fused_timestamp_lru_metadata_update`.
    """
    _validate_fused_timestamp_lru_inputs(
        req_indices,
        topk_indices,
        device_token_pos,
        hit_position_mask,
        device_lru_slots,
        device_lru_slot_stamps,
    )
    probation_age = int(probation_age)
    stamp_max = int(stamp_max)
    if probation_age < 0 or probation_age > stamp_max:
        raise ValueError(
            "probation_age must be in [0, stamp_max], got "
            f"{probation_age} with stamp_max={stamp_max}"
        )
    return torch.ops.npu.fused_timestamp_lru_metadata_update_with_probation(
        req_indices,
        topk_indices,
        device_token_pos,
        hit_position_mask,
        device_lru_slots,
        device_lru_slot_stamps,
        int(max_context_len),
        probation_age,
        stamp_max,
        int(block_dim),
    )


def parallel_lru_metadata_write(
    slot_map: torch.Tensor,
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    victim_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    device_slot_tokens: torch.Tensor,
    max_context_len: int,
    block_dim: int = 0,
) -> None:
    """Apply victim metadata updates across all available AIVs.

    This must run after ``fused_timestamp_lru_metadata_update`` on the same
    stream, or after an explicit dependency on its outputs.
    """
    tensors = {
        "slot_map": slot_map,
        "req_indices": req_indices,
        "topk_indices": topk_indices,
        "victim_slots": victim_slots,
        "miss_counts": miss_counts,
        "device_slot_tokens": device_slot_tokens,
    }
    for name, tensor in tensors.items():
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if topk_indices.dim() != 2 or topk_indices.size(1) != 2048:
        raise ValueError(
            "parallel LRU metadata write requires topk_indices shape "
            f"[batch, 2048], got {tuple(topk_indices.shape)}"
        )
    if victim_slots.shape != topk_indices.shape:
        raise ValueError(
            "victim_slots shape must match topk_indices, got "
            f"{tuple(victim_slots.shape)} and {tuple(topk_indices.shape)}"
        )
    if miss_counts.shape != (topk_indices.size(0),):
        raise ValueError(
            "miss_counts shape must be [batch], got "
            f"{tuple(miss_counts.shape)}"
        )
    if (
        device_slot_tokens.dim() != 2
        or device_slot_tokens.size(1) not in _SUPPORTED_LRU_CACHE_CAPACITIES
    ):
        raise ValueError(
            "device_slot_tokens must have shape "
            "[request_rows, cache_capacity] with cache_capacity in "
            f"{_SUPPORTED_LRU_CACHE_CAPACITIES}, got "
            f"{tuple(device_slot_tokens.shape)}"
        )
    torch.ops.npu.parallel_lru_metadata_write(
        slot_map,
        req_indices,
        topk_indices,
        victim_slots,
        miss_counts,
        device_slot_tokens,
        int(max_context_len),
        int(block_dim),
    )
