import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kda_conv_snapshot_kernel(
    dst_ptr,
    src_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    step_indices_ptr,
    dst_layer_stride,
    dst_req_stride,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    NUM_LAYERS: tl.constexpr,
    SRC_REQ_SIZE: tl.constexpr,
    SRC_STEP_SIZE: tl.constexpr,
    DST_REQ_SIZE: tl.constexpr,
    TAIL_NUMEL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one accepted contiguous conv snapshot per request and layer."""
    pid = tl.program_id(0)
    layer_idx = pid % NUM_LAYERS
    req_idx = pid // NUM_LAYERS
    dst_idx = tl.load(dst_indices_ptr + req_idx).to(tl.int64)
    src_idx = tl.load(src_indices_ptr + req_idx).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + req_idx).to(tl.int64)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < DST_REQ_SIZE)
        & (src_idx >= 0)
        & (src_idx < SRC_REQ_SIZE)
        & (step_idx >= 0)
        & (step_idx < SRC_STEP_SIZE)
    )
    safe_dst_idx = tl.maximum(dst_idx, 0)
    safe_src_idx = tl.maximum(src_idx, 0)
    safe_step_idx = tl.maximum(step_idx, 0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = valid & (offsets < TAIL_NUMEL)
    src_offsets = (
        layer_idx * src_layer_stride
        + safe_src_idx * src_req_stride
        + safe_step_idx * src_step_stride
        + offsets
    )
    dst_offsets = layer_idx * dst_layer_stride + safe_dst_idx * dst_req_stride + offsets
    value = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, value, mask=mask)


def scatter_kda_conv_snapshot(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> bool:
    """Fast path for KDA's contiguous NPU conv-snapshot layout.

    The generic scatter splits Kimi-K3's 6,912-element snapshot into seven
    1,024-element logical tiles per layer, then serializes those tiles over a
    bounded 48-program grid. A single 8,192-element BF16 tile fits in A3 UB
    and needs only one program per request/layer. Unsupported layouts return
    ``False`` so callers can use the generic stride-aware fallback.
    """
    if (
        dst.ndim != 4
        or src.ndim != 5
        or not dst.is_contiguous()
        or not src.is_contiguous()
        or dst.shape[0] != src.shape[0]
        or dst.shape[2:] != src.shape[3:]
    ):
        return False
    num_requests = int(step_indices.shape[0])
    if num_requests == 0:
        return True
    if (
        dst_indices.ndim != 1
        or src_indices.ndim != 1
        or step_indices.ndim != 1
        or dst_indices.shape[0] != num_requests
        or src_indices.shape[0] != num_requests
        or any(
            value.dtype != torch.int32
            for value in (dst_indices, src_indices, step_indices)
        )
    ):
        return False
    tail_numel = int(dst.shape[2] * dst.shape[3])
    block_size = triton.next_power_of_2(tail_numel)
    if block_size > 8192:
        return False
    num_layers = int(dst.shape[0])
    _scatter_kda_conv_snapshot_kernel[(num_requests * num_layers,)](
        dst,
        src,
        dst_indices,
        src_indices,
        step_indices,
        dst.stride(0),
        dst.stride(1),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        NUM_LAYERS=num_layers,
        SRC_REQ_SIZE=src.shape[1],
        SRC_STEP_SIZE=src.shape[2],
        DST_REQ_SIZE=dst.shape[1],
        TAIL_NUMEL=tail_numel,
        BLOCK_SIZE=block_size,
    )
    return True


@triton.jit
def _move_kda_temporal_snapshot_kernel(
    dst_ptr,
    src_ptr,
    dst_indices_ptr,
    src_indices_ptr,
    step_indices_ptr,
    dst_layer_stride,
    dst_req_stride,
    dst_head_stride,
    dst_v_stride,
    dst_k_stride,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    src_head_stride,
    src_v_stride,
    src_k_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    head_dim,
    dim_v,
    dim_k,
    NUM_LAYERS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Copy one KDA temporal snapshot with layer-level parallelism."""
    pid = tl.program_id(0)
    layer_idx = pid % NUM_LAYERS
    req_idx = pid // NUM_LAYERS
    dst_idx = tl.load(dst_indices_ptr + req_idx).to(tl.int64)
    src_idx = tl.load(src_indices_ptr + req_idx).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + req_idx).to(tl.int64)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx >= 0)
        & (step_idx < src_step_size)
    )
    if not valid:
        return

    v_offsets = tl.arange(0, BLOCK_V)
    k_offsets = tl.arange(0, BLOCK_K)
    num_v_tiles = tl.cdiv(dim_v, BLOCK_V)
    num_tiles = head_dim * num_v_tiles
    for tile_idx in tl.range(0, num_tiles):
        head_idx = tile_idx // num_v_tiles
        v_idx = (tile_idx % num_v_tiles) * BLOCK_V + v_offsets
        mask = (v_idx[:, None] < dim_v) & (k_offsets[None, :] < dim_k)
        src_offsets = (
            layer_idx * src_layer_stride
            + src_idx * src_req_stride
            + step_idx * src_step_stride
            + head_idx * src_head_stride
            + v_idx[:, None] * src_v_stride
            + k_offsets[None, :] * src_k_stride
        )
        dst_offsets = (
            layer_idx * dst_layer_stride
            + dst_idx * dst_req_stride
            + head_idx * dst_head_stride
            + v_idx[:, None] * dst_v_stride
            + k_offsets[None, :] * dst_k_stride
        )
        value = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + dst_offsets, value, mask=mask)


def move_kda_temporal_snapshot(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
) -> bool:
    """Fast path for Kimi-K3's production temporal-state layout.

    The generic mover assigns one program to a request, so batch size one
    serializes every KDA layer and head. This path assigns one program per
    request/layer while retaining a runtime loop over 32-row transpose tiles.
    Unsupported layouts return ``False`` so callers can use the generic
    stride-aware fallback.
    """
    if (
        dst.ndim != 5
        or src.ndim != 6
        or dst.shape[0] != src.shape[0]
        or dst.shape[2:] != src.shape[3:]
        or dst.dtype != src.dtype
        or not src.is_contiguous()
    ):
        return False
    num_requests = int(step_indices.shape[0])
    if num_requests == 0:
        return True
    if (
        dst_indices.ndim != 1
        or src_indices.ndim != 1
        or step_indices.ndim != 1
        or dst_indices.shape[0] != num_requests
        or src_indices.shape[0] != num_requests
        or any(
            value.dtype != torch.int32
            for value in (dst_indices, src_indices, step_indices)
        )
    ):
        return False
    dim_k = int(src.shape[-1])
    block_k = triton.next_power_of_2(dim_k)
    if block_k > 128:
        return False
    num_layers = int(src.shape[0])
    _move_kda_temporal_snapshot_kernel[(num_requests * num_layers,)](
        dst,
        src,
        dst_indices,
        src_indices,
        step_indices,
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        dst.stride(3),
        dst.stride(4),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        src.stride(4),
        src.stride(5),
        src.shape[1],
        src.shape[2],
        dst.shape[1],
        src.shape[3],
        src.shape[4],
        dim_k,
        NUM_LAYERS=num_layers,
        BLOCK_V=32,
        BLOCK_K=block_k,
    )
    return True


__all__ = ["move_kda_temporal_snapshot", "scatter_kda_conv_snapshot"]
