"""Public rmsnorm_bias operator with parameter preparation in registered adapters."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.rmsnorm_bias", "auto")
@register_impl("norm.rmsnorm_bias", "triton")
def _triton_impl(
    input,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
):
    import triton

    from ...implementation.triton.norm.rmsnorm_bias import rmsnorm_bias_kernel
    from ...utils.triton_utils import get_device_properties
    from ._rmsnorm_bias_inputs import prepare_inputs

    input, norm_weight, norm_bias, quant_scale, quant_offset = prepare_inputs(
        input, norm_weight, norm_bias, quant_scale, quant_offset
    )
    _, num_vectorcore = get_device_properties()

    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    COL_BLOCK_SIZE = 1024  # 1280 # 1728
    n_rows = min(batch_size, num_vectorcore)

    SCALE = quant_scale is not None
    if SCALE:
        output = torch.empty(
            batch_size, hidden_size, device=input.device, dtype=torch.int8
        )
    else:
        output = torch.empty(
            batch_size, hidden_size, device=input.device, dtype=input.dtype
        )

    if batch_size == 0:
        return output

    rmsnorm_bias_kernel[(n_rows,)](
        input,
        norm_weight,
        norm_bias,
        quant_scale,
        quant_offset,
        output,
        batch_size,
        hidden_size,
        eps,
        BLOCK_SIZE,
        COL_BLOCK_SIZE,
        SCALE,
    )

    return output


@register_impl("norm.rmsnorm_bias", "torch_native")
def _torch_native_impl(
    input, norm_weight, norm_bias, eps, quant_scale=None, quant_offset=None
):
    from ...implementation.torch_native.norm.rmsnorm_bias import rmsnorm_bias
    from ._rmsnorm_bias_inputs import prepare_inputs

    input, norm_weight, norm_bias, quant_scale, quant_offset = prepare_inputs(
        input, norm_weight, norm_bias, quant_scale, quant_offset
    )
    return rmsnorm_bias(input, norm_weight, norm_bias, eps, quant_scale, quant_offset)


@register_operator("norm.rmsnorm_bias")
def rmsnorm_bias(
    input,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
    *,
    impl="auto"
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
            "torch_native". Defaults to "auto"."""
    return get_impl_by_name("norm.rmsnorm_bias", impl)(
        input, norm_weight, norm_bias, eps, quant_scale, quant_offset
    )
