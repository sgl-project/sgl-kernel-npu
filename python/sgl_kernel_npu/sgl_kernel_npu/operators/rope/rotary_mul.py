"""Public rotary_mul operator."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "rope.rotary_mul"
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-transformer")
def _torch_npu_impl(input, r1, r2, rotary_mode="half", rotate=None):
    # The optional rotation matrix was added after the original binding.
    if rotate is not None:
        return torch_npu.npu_rotary_mul(
            input, r1, r2, rotary_mode=rotary_mode, rotate=rotate
        )
    if rotary_mode != "half":
        return torch_npu.npu_rotary_mul(input, r1, r2, rotary_mode=rotary_mode)
    return torch_npu.npu_rotary_mul(input, r1, r2)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(input, r1, r2, rotary_mode="half", rotate=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.rope.rotary_mul import rotary_mul

        _torch_native_fn = rotary_mul
    return _torch_native_fn(input, r1, r2, rotary_mode, rotate)


@register_operator(_OP_NAME)
def rotary_mul(input, r1, r2, rotary_mode="half", rotate=None, *, impl="auto"):
    """Apply RoPE to a floating-point input with an even last dimension.

    r1 and r2 hold broadcastable cosine and sine values. rotary_mode
    controls channel pairing. Newer TorchNPU releases accept a rotate
    matrix on arch22. The torch_native reference covers half and
    interleave modes, with no custom rotation matrix.
    auto selects ops-transformer and returns a new tensor.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, r1, r2, rotary_mode, rotate)
