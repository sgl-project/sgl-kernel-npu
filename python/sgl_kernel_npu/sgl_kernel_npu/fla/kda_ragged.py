import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kda_verify_inputs_kernel(
    qkv_ptr,
    a_ptr,
    b_ptr,
    query_start_loc_ptr,
    dense_qkv_ptr,
    dense_a_ptr,
    dense_b_ptr,
    packed_tokens,
    qkv_stride,
    a_stride,
    b_stride,
    QKV_WIDTH: tl.constexpr,
    A_WIDTH: tl.constexpr,
    B_WIDTH: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dense_row = tl.program_id(0)
    block = tl.program_id(1)
    sequence = dense_row // STEPS
    step = dense_row - sequence * STEPS
    sequence_start = tl.load(query_start_loc_ptr + sequence).to(tl.int64)
    sequence_end = tl.load(query_start_loc_ptr + sequence + 1).to(tl.int64)
    source_row = sequence_start + step
    valid_row = (source_row < sequence_end) & (source_row < packed_tokens)
    offsets = block * BLOCK + tl.arange(0, BLOCK)

    qkv_mask = valid_row & (offsets < QKV_WIDTH)
    qkv = tl.load(
        qkv_ptr + source_row * qkv_stride + offsets,
        mask=qkv_mask,
        other=0.0,
    )
    tl.store(
        dense_qkv_ptr + dense_row * QKV_WIDTH + offsets,
        qkv,
        mask=offsets < QKV_WIDTH,
    )

    a_mask = valid_row & (offsets < A_WIDTH)
    a = tl.load(
        a_ptr + source_row * a_stride + offsets,
        mask=a_mask,
        other=0.0,
    )
    tl.store(
        dense_a_ptr + dense_row * A_WIDTH + offsets,
        a,
        mask=offsets < A_WIDTH,
    )

    b_mask = valid_row & (offsets < B_WIDTH)
    b = tl.load(
        b_ptr + source_row * b_stride + offsets,
        mask=b_mask,
        other=0.0,
    )
    tl.store(
        dense_b_ptr + dense_row * B_WIDTH + offsets,
        b,
        mask=offsets < B_WIDTH,
    )


@triton.jit
def _gather_kda_verify_output_kernel(
    dense_ptr,
    dense_indices_ptr,
    packed_ptr,
    packed_tokens,
    dense_tokens,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    packed_row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dense_row = tl.load(dense_indices_ptr + packed_row).to(tl.int64)
    valid_row = dense_row < dense_tokens
    mask = valid_row & (offsets < WIDTH)
    value = tl.load(
        dense_ptr + dense_row * WIDTH + offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(
        packed_ptr + packed_row * WIDTH + offsets,
        value,
        mask=(packed_row < packed_tokens) & (offsets < WIDTH),
    )


@triton.jit
def _gather_kda_verify_output_norm_kernel(
    dense_ptr,
    dense_indices_ptr,
    gate_ptr,
    weight_ptr,
    packed_ptr,
    packed_tokens,
    dense_tokens,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    packed_row = rows // HEADS
    head = rows - packed_row * HEADS
    offsets = tl.arange(0, BLOCK_D)
    row_mask = rows < packed_tokens * HEADS
    dense_row = tl.load(
        dense_indices_ptr + packed_row, mask=row_mask, other=dense_tokens
    ).to(tl.int64)
    valid_row = row_mask & (dense_row < dense_tokens)
    dim_mask = offsets < HEAD_DIM
    mask = valid_row[:, None] & dim_mask[None, :]

    dense_offsets = (dense_row[:, None] * HEADS + head[:, None]) * HEAD_DIM + offsets[
        None, :
    ]
    value = tl.load(
        dense_ptr + dense_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + rows[:, None] * HEAD_DIM + offsets[None, :],
        mask=row_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=dim_mask, other=0.0).to(tl.float32)
    value = tl.where(mask, value, 0.0)
    rstd = tl.rsqrt(tl.sum(value * value, axis=1) / HEAD_DIM + eps)
    output = value * rstd[:, None] * weight[None, :] * tl.sigmoid(gate)
    tl.store(
        packed_ptr + rows[:, None] * HEAD_DIM + offsets[None, :],
        output,
        mask=row_mask[:, None] & dim_mask[None, :],
    )


def scatter_kda_verify_inputs_npu(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    draft_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map packed ragged KDA inputs to one fixed-width dense layout."""
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(1) != 1:
        raise ValueError("mixed_qkv must be a row-contiguous matrix")
    if a.shape[0] != 1 or b.shape[0] != 1:
        raise ValueError("a and b must have a leading singleton dimension")
    packed_tokens = mixed_qkv.shape[0]
    a_2d = a.squeeze(0).reshape(packed_tokens, -1)
    b_2d = b.squeeze(0).reshape(packed_tokens, -1)
    if a_2d.stride(1) != 1 or b_2d.stride(1) != 1:
        raise ValueError("a and b must be row-contiguous")

    batch_size = query_start_loc.shape[0] - 1
    dense_tokens = batch_size * draft_token_num
    dense_qkv = torch.empty(
        (dense_tokens, mixed_qkv.shape[1]),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    dense_a_2d = torch.empty(
        (dense_tokens, a_2d.shape[1]), dtype=a.dtype, device=a.device
    )
    dense_b_2d = torch.empty(
        (dense_tokens, b_2d.shape[1]), dtype=b.dtype, device=b.device
    )
    block = 1024
    max_width = max(mixed_qkv.shape[1], a_2d.shape[1], b_2d.shape[1])
    grid = (dense_tokens, triton.cdiv(max_width, block))
    _scatter_kda_verify_inputs_kernel[grid](
        mixed_qkv,
        a_2d,
        b_2d,
        query_start_loc,
        dense_qkv,
        dense_a_2d,
        dense_b_2d,
        packed_tokens,
        mixed_qkv.stride(0),
        a_2d.stride(0),
        b_2d.stride(0),
        QKV_WIDTH=mixed_qkv.shape[1],
        A_WIDTH=a_2d.shape[1],
        B_WIDTH=b_2d.shape[1],
        STEPS=draft_token_num,
        BLOCK=block,
    )
    return (
        dense_qkv,
        dense_a_2d.view(1, dense_tokens, *a.shape[2:]),
        dense_b_2d.view(1, dense_tokens, *b.shape[2:]),
    )


def gather_kda_verify_output_npu(
    dense_output: torch.Tensor,
    dense_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Map dense recurrent output back to the packed ragged token order."""
    if dense_output.shape[0] != 1:
        raise ValueError("dense_output must have a leading singleton dimension")
    dense_tokens = dense_output.shape[1]
    dense_2d = dense_output.squeeze(0).reshape(dense_tokens, -1)
    if dense_2d.stride(1) != 1:
        raise ValueError("dense_output must be row-contiguous")
    packed_tokens = dense_token_indices.numel()
    packed_2d = torch.empty(
        (packed_tokens, dense_2d.shape[1]),
        dtype=dense_output.dtype,
        device=dense_output.device,
    )
    block = 1024
    grid = (packed_tokens, triton.cdiv(dense_2d.shape[1], block))
    _gather_kda_verify_output_kernel[grid](
        dense_2d,
        dense_token_indices,
        packed_2d,
        packed_tokens,
        dense_tokens,
        WIDTH=dense_2d.shape[1],
        BLOCK=block,
    )
    return packed_2d.view(1, packed_tokens, *dense_output.shape[2:])


def gather_kda_verify_output_norm_npu(
    dense_output: torch.Tensor,
    dense_token_indices: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Gather ragged KDA output and apply sigmoid-gated RMSNorm in one kernel."""
    if dense_output.ndim != 4 or dense_output.shape[0] != 1:
        raise ValueError("dense_output must have shape [1, dense_tokens, H, D]")
    dense_tokens, heads, head_dim = dense_output.shape[1:]
    if dense_output.stride(-1) != 1:
        raise ValueError("dense_output must be contiguous in the head dimension")
    packed_tokens = dense_token_indices.numel()
    if gate.numel() != packed_tokens * heads * head_dim or gate.stride(-1) != 1:
        raise ValueError("gate must be a row-contiguous packed [tokens, H * D] tensor")
    if weight.numel() != head_dim or not weight.is_contiguous():
        raise ValueError("weight must be contiguous with one value per head dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")

    packed = torch.empty(
        (1, packed_tokens, heads, head_dim),
        dtype=dense_output.dtype,
        device=dense_output.device,
    )
    block_t = 32
    _gather_kda_verify_output_norm_kernel[
        (triton.cdiv(packed_tokens * heads, block_t),)
    ](
        dense_output,
        dense_token_indices,
        gate,
        weight,
        packed,
        packed_tokens,
        dense_tokens,
        eps,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_T=block_t,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return packed
