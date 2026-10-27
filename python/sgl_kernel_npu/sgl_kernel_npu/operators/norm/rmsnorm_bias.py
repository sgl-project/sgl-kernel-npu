"""Public RMSNorm+bias operator demonstrating registration and dispatch."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


# Importing this module registers these two functions, without importing Triton.
@register_impl("norm.rmsnorm_bias", "auto")
@register_impl("norm.rmsnorm_bias", "triton")
def _triton_impl(input, norm_weight, norm_bias, eps, quant_scale, quant_offset):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.rmsnorm_bias import rmsnorm_bias

        _triton_fn = rmsnorm_bias

    return _triton_fn(input, norm_weight, norm_bias, eps, quant_scale, quant_offset)


@register_impl("norm.rmsnorm_bias", "torch_native")
def _torch_native_impl(input, norm_weight, norm_bias, eps, quant_scale, quant_offset):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.rmsnorm_bias import rmsnorm_bias

        _torch_native_fn = rmsnorm_bias

    return _torch_native_fn(
        input, norm_weight, norm_bias, eps, quant_scale, quant_offset
    )


@register_operator("norm.rmsnorm_bias")
def rmsnorm_bias(
    input,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
    *,
    impl="auto",
):
    """The default impl="auto" selects Triton.

    Apply RMS normalization over H, followed by per-channel weight and bias,
    with optional int8 quantization.

    Args:
        input (torch.Tensor): Input of shape [B, H]; B rows, H features.
        norm_weight (torch.Tensor): Multiplicative weight of shape [H].
        norm_bias (torch.Tensor): Additive bias of shape [H].
        eps (float): Stability term added to the mean square.
        quant_scale (torch.Tensor | None): Quantization multiplier of shape
            [H]; None disables quantization.
        quant_offset (torch.Tensor | None): Quantization offset of shape [H];
            required when quant_scale is provided, otherwise ignored.
        impl (str): Implementation name: "auto", "triton", or
            "torch_native". Defaults to "auto".
    """
    impl_fn = get_impl_by_name("norm.rmsnorm_bias", impl)
    return impl_fn(input, norm_weight, norm_bias, eps, quant_scale, quant_offset)
