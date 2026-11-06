"""Bit-exact numeric helpers shared by the Triton implementations."""

import triton
import triton.language as tl


@triton.jit
def cast_rne_bf16_to_fp32(value):
    """Round an FP32 value to BF16 precision, round-to-nearest-even.

    On Ascend ``to(tl.bfloat16)`` lowers to one ``vcast``; emulating the same
    rounding with integer operations costs three vector ops per element and
    dominates bandwidth-bound kernels. Both forms round to nearest even, so the
    results are identical.
    """
    return value.to(tl.bfloat16).to(tl.float32)
