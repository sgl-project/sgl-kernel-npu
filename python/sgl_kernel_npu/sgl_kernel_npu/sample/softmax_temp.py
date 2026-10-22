"""DSpark temperature probability construction, ported from SGLang.

Temperatures are per request (row // rows_per_request), output is FP32.
Zero-temperature greedy requests must be routed by the caller.
"""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _softmax_temp_kernel(
    logits_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    rows_per_request,
    logits_row_stride,
    num_rows,
    BLOCK_V: tl.constexpr,
):
    for row in range(tl.program_id(0), num_rows, tl.num_programs(0)):
        temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
        base = logits_ptr + row.to(tl.int64) * logits_row_stride
        out_base = out_ptr + row.to(tl.int64) * vocab

        row_max = -float("inf")
        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
            x = x / temp
            row_max = tl.maximum(row_max, tl.max(x, axis=0))

        sum_exp = 0.0
        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
            x = x / temp
            e = tl.exp(x - row_max)
            e = tl.where(vmask, e, 0.0)
            sum_exp += tl.sum(e, axis=0)

        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
            x = x / temp
            e = tl.exp(x - row_max)
            tl.store(out_base + offs, e / sum_exp, mask=vmask)


def softmax_temp_triton(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    assert logits.ndim == 2 and logits.stride(1) == 1
    assert logits.device == temperatures.device
    assert rows_per_request > 0 and logits.shape[1] > 0
    num_rows, vocab = logits.shape[0], logits.shape[-1]
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temperatures = temperatures.reshape(bs).to(torch.float32).contiguous()
    out = torch.empty((num_rows, vocab), dtype=torch.float32, device=logits.device)
    if num_rows == 0:
        return out
    BLOCK_V = min(4096, triton.next_power_of_2(vocab))
    _softmax_temp_kernel[(min(get_device_properties()[1], num_rows),)](
        logits,
        temperatures,
        out,
        vocab,
        rows_per_request,
        logits.stride(0),
        num_rows,
        BLOCK_V=BLOCK_V,
    )
    return out
