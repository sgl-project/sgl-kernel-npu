"""DSV4 rope operators with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("dsv4.inplace_partial_rotary_mul", "auto")
@register_impl("dsv4.inplace_partial_rotary_mul", "cann_ops_transformer")
def _cann_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    from ...implementation.cann_ops_transformer.dsv4 import inplace_partial_rotary_mul

    return inplace_partial_rotary_mul(x, cos, sin, **kwargs)


@register_impl("dsv4.inplace_partial_rotary_mul", "torch_native")
def _torch_inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    from ...implementation.torch_native.dsv4.rope import inplace_partial_rotary_mul

    return inplace_partial_rotary_mul(x, cos, sin, **kwargs)


@register_operator("dsv4.inplace_partial_rotary_mul")
def inplace_partial_rotary_mul(x, cos, sin, *, impl="auto", **kwargs):
    """Apply partial RoPE in place using fp32 intermediate arithmetic.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.inplace_partial_rotary_mul", impl)(
        x, cos, sin, **kwargs
    )
