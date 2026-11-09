"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_short_conv_triton_fn = None
_short_conv_torch_fn = None


@register_impl("mamba.short_conv", "auto", supported_generations={Generation.ARCH22})
@register_impl("mamba.short_conv", "triton", supported_generations={Generation.ARCH22})
def _short_conv_triton(conv_input, weight, dilation, *, single_token=False):
    global _short_conv_triton_fn
    if _short_conv_triton_fn is None:
        from ...implementation.triton.mamba.short_conv import short_conv

        _short_conv_triton_fn = short_conv
    return _short_conv_triton_fn(
        conv_input, weight, dilation, single_token=single_token
    )


@register_impl(
    "mamba.short_conv", "torch_native", supported_generations={Generation.ARCH22}
)
def _short_conv_torch(conv_input, weight, dilation, *, single_token=False):
    global _short_conv_torch_fn
    if _short_conv_torch_fn is None:
        from ...implementation.torch_native.mamba.short_conv import short_conv

        _short_conv_torch_fn = short_conv
    return _short_conv_torch_fn(conv_input, weight, dilation, single_token=single_token)


@register_operator("mamba.short_conv")
def short_conv(conv_input, weight, dilation, *, single_token=False, impl="auto"):
    """Depthwise dilation-3 PLE convolution. Input [B,10240,9+W], W=1..4; matching BF16/FP16 weight [10240,1,4]. Returns contiguous [B,W,10240], or [B,10240] for single_token=True. auto selects Triton.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("mamba.short_conv", impl)(
        conv_input, weight, dilation, single_token=single_token
    )
