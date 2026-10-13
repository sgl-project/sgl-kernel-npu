"""Public RMSNorm+bias operator demonstrating registration and dispatch."""

from ..utils.registry import get_impl, register_impl


# Importing this module registers these two functions, without importing Triton.
@register_impl("norm.rmsnorm_bias", "triton")
def _triton_impl(input, norm_weight, norm_bias, eps, quant_scale, quant_offset):
    from ...implemention.triton.norm.rmsnorm_bias import rmsnorm_bias

    return rmsnorm_bias(
        input, norm_weight, norm_bias, eps, quant_scale, quant_offset
    )


@register_impl("norm.rmsnorm_bias", "torch_native")
def _torch_native_impl(input, norm_weight, norm_bias, eps, quant_scale, quant_offset):
    from ...implemention.torch_native.norm.rmsnorm_bias import rmsnorm_bias

    return rmsnorm_bias(
        input, norm_weight, norm_bias, eps, quant_scale, quant_offset
    )


def rmsnorm_bias(
    input,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
    *,
    implementation="triton",
):
    """Compute RMSNorm(input) * norm_weight + norm_bias.

    Inputs are forwarded to the selected implementation without validation.
    The default preserves the original Triton implementation.
    torch_native currently supports only the non-quantized operation.
    This example provides explicit selection only (no auto policy).
    """
    impl = get_impl("norm.rmsnorm_bias", implementation)
    return impl(input, norm_weight, norm_bias, eps, quant_scale, quant_offset)
