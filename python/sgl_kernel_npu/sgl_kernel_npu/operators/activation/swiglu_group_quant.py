import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "activation.swiglu_group_quant", "auto", supported_generations=[Generation.ARCH35]
)
@register_impl(
    "activation.swiglu_group_quant",
    "ascendc",
    supported_generations=[Generation.ARCH35],
)
def _ascendc(**kwargs):
    return torch.ops.npu.swiglu_group_quant(**kwargs)


@register_impl("activation.swiglu_group_quant", "torch_native")
def _native(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.swiglu_group_quant import (
            swiglu_group_quant,
        )

        _torch_native_fn = swiglu_group_quant
    return _torch_native_fn(**kwargs)


@register_operator("activation.swiglu_group_quant")
def swiglu_group_quant(
    x,
    topk_weight=None,
    group_index=None,
    dst_type=None,
    quant_mode=1,
    group_size=128,
    round_scale=False,
    ue8m0_scale=False,
    output_origin=False,
    group_list_type=0,
    clamp_value=0.0,
    *,
    impl="auto",
):
    """Apply SwiGLU and quantize expert activations; auto selects AscendC on arch35.

    Computes SiLU(gate) * up with optional clamp and routing weights. Mode 1
    uses FP32 scales per 128 values; mode 2 uses E8M0 scales per 32 values;
    mode 3 uses a fixed scale reference of 448 with optional power-of-two scales.

    Args:
        x: Contiguous FP16/BF16 tensor [..., 2*C]. The first half contains gate and
            the second up. N is the product of the leading dimensions. The last
            dimension must be divisible by 256 in modes 1/3 or 128 in mode 2.
        topk_weight: Optional contiguous FP32 tensor [N] on x's device, containing
            one routing weight per flattened row. None or empty means no weight.
        group_index: Optional contiguous int64 tensor [E] on x's device, containing
            nonnegative expert counts. Process min(N, sum(counts)) leading rows;
            None or empty processes all rows. Remaining output rows are unspecified.
        dst_type: Output dtype, torch.float8_e4m3fn (default) or torch.float8_e5m2.
        quant_mode: Integer mode, default 1. Modes 1/3 use 128 values per scale;
            mode 2 uses MX quantization with 32 values per scale.
        group_size: Integer compatibility parameter, default 128; does not change
            the fixed group widths selected by quant_mode.
        round_scale: Boolean, default False. Round mode-3 scales up to powers of two.
        ue8m0_scale: Boolean, default False. Store mode-3 scales in E8M0; use with
            round_scale=True. Mode 2 always returns E8M0 scales.
        output_origin: Boolean, default False. Write unquantized activations in mode 3.
        group_list_type: Integer count-list format; use 0 (default).
        clamp_value: Float, default 0.0 (disabled). A positive value bounds gate
            above and up on both sides before activation and optional weighting.
        impl: Keyword-only implementation: auto (default), ascendc on arch35,
            or torch_native. Auto does not fall back on unsupported hardware.

    Returns:
        Tuple (y, scale, y_origin). y has shape [..., C] and dst_type. scale has
        shape [..., C/64, 2] and E8M0 dtype in mode 2; modes 1/3 use [..., C/128]
        and FP32, except mode 3 with ue8m0_scale=True uses E8M0. y_origin has x's
        dtype and shape [..., C], and is defined only for processed rows in mode 3
        with output_origin=True. Outputs use x's device. Inputs are not modified.
    """
    return get_impl_by_name("activation.swiglu_group_quant", impl)(
        x=x,
        topk_weight=topk_weight,
        group_index=group_index,
        dst_type=dst_type,
        quant_mode=quant_mode,
        group_size=group_size,
        round_scale=round_scale,
        ue8m0_scale=ue8m0_scale,
        output_origin=output_origin,
        group_list_type=group_list_type,
        clamp_value=clamp_value,
    )
