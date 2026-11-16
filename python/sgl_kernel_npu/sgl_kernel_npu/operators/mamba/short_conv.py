"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_short_conv_triton_fn = None
_short_conv_torch_fn = None


@register_impl(
    "mamba.short_conv",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "mamba.short_conv",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _short_conv_triton(conv_input, weight, dilation, *, single_token=False):
    global _short_conv_triton_fn
    if _short_conv_triton_fn is None:
        from ...implementation.triton.mamba.short_conv import short_conv

        _short_conv_triton_fn = short_conv
    return _short_conv_triton_fn(
        conv_input, weight, dilation, single_token=single_token
    )


@register_impl(
    "mamba.short_conv",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _short_conv_torch(conv_input, weight, dilation, *, single_token=False):
    global _short_conv_torch_fn
    if _short_conv_torch_fn is None:
        from ...implementation.torch_native.mamba.short_conv import short_conv

        _short_conv_torch_fn = short_conv
    return _short_conv_torch_fn(conv_input, weight, dilation, single_token=single_token)


@register_operator("mamba.short_conv")
def short_conv(conv_input, weight, dilation, *, single_token=False, impl="auto"):
    """Apply depthwise convolution with dilation 3 to PLE inputs.

    conv_input is contiguous [B,10240,9+W], W=1..4. weight is contiguous
    [10240,1,4] on the same NPU, with the same BF16 or FP16 dtype.
    Integer dilation must be 3. Boolean single_token defaults to False;
    True requires W=1. Return [B,W,10240], or [B,10240] when single_token
    is True, in the input dtype. Inputs are unchanged. impl defaults to auto,
    selecting Triton.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("mamba.short_conv", impl)(
        conv_input, weight, dilation, single_token=single_token
    )
