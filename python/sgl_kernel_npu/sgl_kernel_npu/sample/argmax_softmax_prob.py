"""Single-pass fused argmax + softmax-probability-of-argmax kernel.

Confidence-driven decoders (diffusion LLMs in particular) need, per row, the
argmax token and the softmax probability of that token -- the score compared
against an unmask threshold. Done portably that is a separate max / sub / exp /
sum / gather chain over ``[rows, vocab]``, reading the logits five to six times
and materializing full-width temporaries.

This kernel reads each logit once, carrying running ``(max, argmax, sumexp)`` in
registers via the online-softmax recurrence, and returns ``(argmax_id, 1/sumexp)``:
the argmax token's logit is the running max, so its ``exp(max - max)`` is 1 and
its softmax probability is exactly the reciprocal of the sum. Nothing of shape
``[rows, vocab]`` is ever written.

Measured on Ascend 910B3 at vocab 157184, bf16 logits, against an fp32
``argmax`` + ``softmax`` + ``gather`` reference: argmax identical on every row,
probability max relative error 3.5e-7, and no decision flips at a 0.5 threshold.
Against a chunked in-framework reduction: 6.6x at 32 rows (launch-bound) and
2.7x at 4096 rows.
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _argmax_prob_kernel(
    logits_ptr,  # [B, V]
    argmax_ptr,  # [B] int64
    prob_ptr,  # [B] float32
    B,
    V,
    stride_b,
    BLOCK_V: tl.constexpr,
):
    row_pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    for row in tl.range(row_pid, B, n_prog):
        base = row * stride_b
        m = tl.full((), float("-inf"), tl.float32)
        arg = tl.zeros((), tl.int64)
        s = tl.zeros((), tl.float32)
        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            chunk_max = tl.max(x, axis=0)
            new_m = tl.maximum(m, chunk_max)
            s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
            chunk_arg = tl.argmax(x, axis=0).to(tl.int64) + start
            # Strict >: a tie keeps the earlier index, matching torch.argmax.
            arg = tl.where(chunk_max > m, chunk_arg, arg)
            m = new_m
        tl.store(argmax_ptr + row, arg)
        tl.store(prob_ptr + row, 1.0 / s)


@lru_cache(maxsize=1)
def _num_cores() -> int:
    return get_device_properties()[1]


# Vocab tile budget per program, in bytes. 32 KiB is the widest tile that still
# compiles for a 4-byte dtype on 910B3 (64 KiB overflows the unified buffer once
# auto-multi-buffering claims its share); 2-byte dtypes get twice the elements.
_TILE_BYTES = 32 * 1024


def argmax_softmax_prob_fused(
    logits: torch.Tensor, block_v: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row argmax id and the softmax probability of that argmax token.

    ``logits``: ``[B, V]``, any float dtype, last dim contiguous (an arbitrary
    row stride is fine, so a vocab-truncated view costs no copy). Returns
    ``(argmax_ids [B] int64, prob [B] float32)``. Accumulates in fp32; never
    materializes a ``[B, V]`` temporary.

    ``block_v`` overrides the vocab tile width; the default derives it from the
    logits dtype so the tile stays within the unified buffer.
    """
    assert logits.dim() == 2 and logits.stride(1) == 1
    B, V = logits.shape
    argmax = torch.empty(B, dtype=torch.int64, device=logits.device)
    prob = torch.empty(B, dtype=torch.float32, device=logits.device)
    grid = (min(_num_cores(), B),)
    if block_v is None:
        block_v = _TILE_BYTES // logits.element_size()
    # A tile wider than the vocab only wastes unified buffer (and overflows it
    # once the single-iteration loop is unrolled).
    block_v = min(block_v, triton.next_power_of_2(V))
    _argmax_prob_kernel[grid](
        logits, argmax, prob, B, V, logits.stride(0), BLOCK_V=block_v
    )
    return argmax, prob
