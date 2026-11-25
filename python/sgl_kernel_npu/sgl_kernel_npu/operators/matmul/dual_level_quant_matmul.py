"""Multiply dual-level MXFP4 matrices on arch35."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.dual_level_quant_matmul"
_native_impl = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "ops-nn", supported_generations={Generation.ARCH35})
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_dual_level_quant_matmul(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native", supported_generations={Generation.ARCH35})
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.matmul.torch_npu_quant_matmul import (
            dual_level_quant_matmul,
        )

        _native_impl = dual_level_quant_matmul
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def dual_level_quant_matmul(
    x1,
    x2,
    x1_level0_scale,
    x2_level0_scale,
    x1_level1_scale,
    x2_level1_scale,
    *,
    impl="auto",
    **kwargs,
):
    """Multiply dual-level MXFP4 matrices on arch35.

    x1 and x2 are packed matrices. The separate level-0 and level-1 scales
    correspond to each operand. Pass bias and output_dtype as TorchNPU keywords.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(
        x1,
        x2,
        x1_level0_scale,
        x2_level0_scale,
        x1_level1_scale,
        x2_level1_scale,
        **kwargs,
    )
