"""Public interleave_rope operator."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "rope.interleave_rope"
_torch_native_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-transformer")
def _torch_npu_impl(x, cos, sin):
    return torch_npu.npu_interleave_rope(x, cos, sin)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(x, cos, sin):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.rope.interleave_rope import interleave_rope

        _torch_native_fn = interleave_rope
    return _torch_native_fn(x, cos, sin)


@register_operator(_OP_NAME)
def interleave_rope(x, cos, sin, *, impl="auto"):
    """Deinterleave even and odd channels and apply half-rotation RoPE.

    x is [B, N, S, D]. cos and sin broadcast over heads and, when
    their sequence dimension is one, over sequence positions. Returns
    the rotated output in half-split ordering. auto selects ops-transformer.
    """
    return get_impl_by_name(_OP_NAME, impl)(x, cos, sin)
