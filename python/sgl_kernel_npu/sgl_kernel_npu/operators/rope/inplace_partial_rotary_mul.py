"""Public inplace_partial_rotary_mul operator with registered implementations."""

from ..utils._cann_ops_transformer import get_op
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("rope.inplace_partial_rotary_mul", "auto")
@register_impl("rope.inplace_partial_rotary_mul", "cann_ops_transformer")
def _cann_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    get_op("inplace_partial_rotary_mul")(x, cos, sin, **kwargs)
    return x


@register_impl("rope.inplace_partial_rotary_mul", "torch_native")
def _torch_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    from ...implementation.torch_native.rope.inplace_partial_rotary_mul import (
        inplace_partial_rotary_mul,
    )

    return inplace_partial_rotary_mul(x, cos, sin, **kwargs)


@register_operator("rope.inplace_partial_rotary_mul")
def inplace_partial_rotary_mul(x, cos, sin, *, impl="auto", **kwargs):
    """Apply partial RoPE in place using fp32 intermediate arithmetic.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("rope.inplace_partial_rotary_mul", impl)(
        x, cos, sin, **kwargs
    )
