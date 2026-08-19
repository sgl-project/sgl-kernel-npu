import torch
import triton
import triton.language as tl


@triton.jit
def _select_local_top1_after_add_kernel(
    base_ptr,
    bias_ptr,
    output_ptr,
    B,
    V: tl.constexpr,
    vocab_offset: tl.constexpr,
    base_stride,
    bias_stride,
    BLOCK_V: tl.constexpr,
):
    row_pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    for row in tl.range(row_pid, B, n_prog):
        best_value = tl.full((), float("-inf"), tl.float32)
        best_index = tl.zeros((), tl.int64)
        for start in range(0, V, BLOCK_V):
            offsets = start + tl.arange(0, BLOCK_V)
            mask = offsets < V
            base = tl.load(
                base_ptr + row * base_stride + offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            bias = tl.load(
                bias_ptr + row * bias_stride + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            # F.linear and the base logits are BF16 on K3. Match eager's BF16
            # add rounding before comparing values, then reduce in FP32.
            logits = (base + bias).to(
                base_ptr.dtype.element_ty, fp_downcast_rounding="rtne"
            ).to(tl.float32)
            chunk_value = tl.max(logits, axis=0)
            chunk_index = tl.argmax(logits, axis=0).to(tl.int64) + start
            # Strict comparison preserves the first (smallest) index on ties.
            best_index = tl.where(chunk_value > best_value, chunk_index, best_index)
            best_value = tl.maximum(best_value, chunk_value)
        tl.store(output_ptr + row * 2, best_value)
        tl.store(output_ptr + row * 2 + 1, (best_index + vocab_offset).to(tl.float32))


@triton.jit
def _select_global_top1_kernel(
    candidates_ptr,
    output_ptr,
    TP_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_TP: tl.constexpr,
):
    row = tl.program_id(0)
    rank_offsets = tl.arange(0, BLOCK_TP)
    rank_mask = rank_offsets < TP_SIZE
    row_base = row * TP_SIZE * 2
    values = tl.load(
        candidates_ptr + row_base + rank_offsets * 2,
        mask=rank_mask,
        other=float("-inf"),
    )
    token_ids = tl.load(
        candidates_ptr + row_base + rank_offsets * 2 + 1,
        mask=rank_mask,
        other=VOCAB_SIZE,
    ).to(tl.int32)
    best_value = tl.max(values, axis=0)
    best_token = tl.min(
        tl.where(values == best_value, token_ids, VOCAB_SIZE), axis=0
    )
    tl.store(output_ptr + row, best_token.to(tl.int64))


def select_global_top1_npu(
    candidates: torch.Tensor, *, vocab_size: int
) -> torch.Tensor:
    batch_size, tp_size, pair_size = candidates.shape
    assert pair_size == 2 and candidates.is_contiguous()
    output = torch.empty(batch_size, dtype=torch.long, device=candidates.device)
    _select_global_top1_kernel[(batch_size,)](
        candidates,
        output,
        TP_SIZE=tp_size,
        VOCAB_SIZE=vocab_size,
        BLOCK_TP=triton.next_power_of_2(tp_size),
    )
    return output


def select_local_top1_after_add_npu(
    base: torch.Tensor,
    bias: torch.Tensor,
    *,
    vocab_offset: int,
) -> torch.Tensor:
    """Fuse BF16 logits add and local top-1 candidate construction.

    Both inputs are ``[batch, local_vocab]`` with a contiguous last dimension;
    row strides may differ. The returned FP32 pair is ``(value, global_id)``
    and can be passed directly to the TP candidate all-gather.
    """
    if (
        base.ndim != 2
        or bias.ndim != 2
        or base.shape != bias.shape
        or base.dtype != bias.dtype
        or base.stride(1) != 1
        or bias.stride(1) != 1
    ):
        raise ValueError("base and bias must be matching row-contiguous matrices")
    if base.dtype != torch.bfloat16:
        raise ValueError("base and bias must use bfloat16")
    batch_size, local_vocab_size = base.shape
    output = torch.empty((batch_size, 2), dtype=torch.float32, device=base.device)
    block_v = min(2048, triton.next_power_of_2(local_vocab_size))
    _select_local_top1_after_add_kernel[(batch_size,)](
        base,
        bias,
        output,
        batch_size,
        V=local_vocab_size,
        vocab_offset=vocab_offset,
        base_stride=base.stride(0),
        bias_stride=bias.stride(0),
        BLOCK_V=block_v,
    )
    return output
