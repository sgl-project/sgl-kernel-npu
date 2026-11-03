import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


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
    from ...implementation.torch_native.activation.swiglu_group_quant import (
        swiglu_group_quant,
    )

    return swiglu_group_quant(**kwargs)


@register_impl("activation.swiglu_group_quant", "cpu")
def _cpu(**kwargs):
    from ...implementation.cpu.activation.swiglu_group_quant import swiglu_group_quant

    return swiglu_group_quant(**kwargs)


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
    """Apply SwiGLU and quantize expert activations.

    Args:
        x: Contiguous FP16/BF16 gate and up projections concatenated on the last axis.
        topk_weight: Optional float32 routing weight per row.
        group_index: Optional int64 counts per expert. Only the leading sum of counts is processed.
        dst_type: FP8 E4M3FN (default) or E5M2.
        quant_mode: 1 for group quantization, 2 for MX, 3 for per-token FP8 blocks.
        group_size: Accepted for compatibility. Modes 1/3 use 128 values per scale, MX uses 32.
        round_scale: Use power-of-two scales in mode 3.
        ue8m0_scale: Store mode 3 rounded scales in E8M0. Use with round_scale.
        output_origin: Populate the unquantized output in mode 3.
        group_list_type: 0 for per-expert counts.
        clamp_value: Nonzero limit for gate (upper bound) and up (both bounds).
        impl: auto or ascendc on A5, torch_native, or cpu.

    Returns:
        (y, scale, y_origin). y and y_origin have half the input width.
        Only mode 3 with output_origin=True writes y_origin.
    """
    if x.ndim < 1 or not x.is_contiguous():
        raise RuntimeError("x must be contiguous with at least one dimension")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("x should be FLOAT16 or BFLOAT16")
    if quant_mode not in (1, 2, 3):
        raise RuntimeError("Unsupported quant mode")
    dst_type = torch.float8_e4m3fn if dst_type is None else dst_type
    if dst_type not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise RuntimeError("dst_type must be Float8_e4m3fn or Float8_e5m2")
    alignment = 128 if quant_mode == 2 else 256
    if x.size(-1) % alignment:
        raise RuntimeError(f"last dimension must be divisible by {alignment}")
    if x.numel() and group_list_type != 0:
        raise RuntimeError("swiglu_group_quant: tiling failed for group_list_type")
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
