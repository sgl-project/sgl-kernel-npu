"""Public inplace_partial_rotary_mul operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import cann_backend, require_op


@register_impl("rope.inplace_partial_rotary_mul", "auto")
def _auto_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    return get_impl_by_name("rope.inplace_partial_rotary_mul", cann_backend())(
        x, cos, sin, **kwargs
    )


@register_impl("rope.inplace_partial_rotary_mul", "cann_ops_transformer")
def _cann_ops_transformer_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    op = require_op(
        "cann_ops_transformer",
        "inplace_partial_rotary_mul",
        backend="cann_ops_transformer",
    )
    op(x, cos, sin, **kwargs)
    return x


@register_impl("rope.inplace_partial_rotary_mul", "cann_9_1")
def _cann_9_1_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    op = require_op("custom", "inplace_partial_rotary_mul", backend="cann_9_1")
    op(x, cos, sin, **kwargs)
    return x


@register_impl("rope.inplace_partial_rotary_mul", "torch_native")
def _torch_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    from ...implementation.torch_native.rope.inplace_partial_rotary_mul import (
        inplace_partial_rotary_mul,
    )

    return inplace_partial_rotary_mul(x, cos, sin, **kwargs)


@register_operator("rope.inplace_partial_rotary_mul")
def inplace_partial_rotary_mul(
    x, cos, sin, *, rotary_mode="interleave", partial_slice=None, impl="auto"
):
    """Apply partial RoPE in place using fp32 intermediate arithmetic.

    impl: auto (SDK-selected), cann_9_1, cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("rope.inplace_partial_rotary_mul", impl)(
        x,
        cos,
        sin,
        rotary_mode=rotary_mode,
        partial_slice=[0, x.shape[-1]] if partial_slice is None else partial_slice,
    )
