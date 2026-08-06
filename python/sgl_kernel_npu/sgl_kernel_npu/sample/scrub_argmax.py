"""Fused scrubbed argmax for dLLM decoding on Ascend NPU."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _scrub_argmax_kernel(
    logits_ptr,
    scrub_ptr,
    num_rows,
    vocab_size,
    row_stride,
    delete_token_id: tl.constexpr,
    split_token_id: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row_pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row in tl.range(row_pid, num_rows, num_programs):
        base = row * row_stride
        running_max = tl.full((), float("-inf"), tl.float32)
        running_argmax = tl.zeros((), tl.int64)
        for start in range(0, vocab_size, BLOCK_V):
            offsets = start + tl.arange(0, BLOCK_V)
            valid = offsets < vocab_size
            values = tl.load(
                logits_ptr + base + offsets,
                mask=valid,
                other=float("-inf"),
            )
            values = tl.where(
                (offsets != delete_token_id) & (offsets != split_token_id),
                values,
                float("-inf"),
            )
            chunk_max = tl.max(values, axis=0)
            chunk_argmax = tl.argmax(values, axis=0).to(tl.int64) + start
            running_argmax = tl.where(
                chunk_max > running_max,
                chunk_argmax,
                running_argmax,
            )
            running_max = tl.maximum(running_max, chunk_max)
        tl.store(scrub_ptr + row, running_argmax)


def _num_programs(num_rows: int) -> int:
    _, num_cores = get_device_properties()
    return min(num_cores, num_rows)


def scrub_argmax_fused(
    logits: torch.Tensor,
    delete_token_id: int,
    split_token_id: int,
    block_v: int = 8192,
) -> torch.Tensor:
    """Return each row's best token excluding DELETE and SPLIT."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {logits.dim()}D")

    num_rows, vocab_size = logits.shape
    logits = logits.contiguous()
    scrub = torch.empty(num_rows, dtype=torch.int64, device=logits.device)
    _scrub_argmax_kernel[(_num_programs(num_rows),)](
        logits,
        scrub,
        num_rows,
        vocab_size,
        logits.stride(0),
        delete_token_id=delete_token_id,
        split_token_id=split_token_id,
        BLOCK_V=block_v,
        multibuffer=False,
    )
    return scrub
