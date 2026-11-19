"""Multiply dual-level MXFP4 matrices on arch35."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.dual_level_quant_matmul"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "torch_npu", supported_generations={Generation.ARCH35})
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_dual_level_quant_matmul

        _torch_npu_fn = npu_dual_level_quant_matmul
    return _torch_npu_fn(*args, **kwargs)


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
    Both auto and torch_npu use the installed TorchNPU implementation.
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
