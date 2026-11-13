"""Bit-exact ``silu(a) * b`` for Ascend NPUs.

The eager chain is two kernels, ``silu`` and ``mul``, each rounding its BF16
result. This kernel performs both roundings explicitly in one pass: the silu
product is rounded to BF16 before the multiply, and the store rounds the final
product.
"""

import torch
import triton
import triton.language as tl

from ...utils.triton_utils import get_device_properties
from ..diffusion_utils import cast_rne_bf16_to_fp32

# BF16 elements per program iteration. Two inputs and one output are live in
# UB together with the FP32 intermediates, which keeps one tile well inside the
# 192 KB budget.
DEFAULT_TILE = 4096
# The kernels index with 32-bit offsets; larger tensors need a different layout.
MAX_NUMEL = 2**31


@triton.jit(do_not_specialize=["num_tiles"])
def _silu_mul_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    num_tiles,
    NUM_PROGS: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, TILE)
    for tile in range(pid, num_tiles, NUM_PROGS):
        offs = tile * TILE + cols
        a = tl.load(a_ptr + offs).to(tl.float32)
        b = tl.load(b_ptr + offs).to(tl.float32)
        # eager rounds silu to BF16 before the multiply
        s = cast_rne_bf16_to_fp32(a * tl.sigmoid(a))
        tl.store(out_ptr + offs, s * b)


@triton.jit(do_not_specialize=["start", "numel"])
def _silu_mul_tail_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    start,
    numel,
    TILE: tl.constexpr,
):
    offs = start + tl.arange(0, TILE)
    mask = offs < numel
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = cast_rne_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + offs, s * b, mask=mask)


def silu_mul(a, b):
    """Compute ``silu(a) * b`` with the eager BF16 rounding boundaries.

    ``a`` and ``b`` must be contiguous BF16 tensors of the same shape on one
    NPU. The fused path is always used; inputs outside that contract are
    rejected before launch.
    """
    if a.device.type != "npu" or b.device != a.device:
        raise ValueError("a and b must be on the same NPU")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("silu_mul expects BF16 inputs")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("silu_mul expects contiguous inputs")
    numel = a.numel()
    if not 0 < numel < MAX_NUMEL:
        raise ValueError("silu_mul expects 0 < numel < 2**31")

    out = torch.empty_like(a)
    _, num_vectorcore = get_device_properties()
    num_tiles, tail = divmod(numel, DEFAULT_TILE)
    if num_tiles:
        progs = min(num_tiles, num_vectorcore)
        _silu_mul_kernel[(progs,)](
            out,
            a,
            b,
            num_tiles,
            NUM_PROGS=progs,
            TILE=DEFAULT_TILE,
        )
    if tail:
        _silu_mul_tail_kernel[(1,)](
            out,
            a,
            b,
            num_tiles * DEFAULT_TILE,
            numel,
            TILE=triton.next_power_of_2(tail),
        )
    return out
