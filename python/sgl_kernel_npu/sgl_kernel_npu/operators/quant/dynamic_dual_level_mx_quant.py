"""Dynamically quantize to dual-level MXFP4 on arch35."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "quant.dynamic_dual_level_mx_quant"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "torch_npu", supported_generations={Generation.ARCH35})
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_dynamic_dual_level_mx_quant

        _torch_npu_fn = npu_dynamic_dual_level_mx_quant
    return _torch_npu_fn(*args, **kwargs)


@register_operator(_OP_NAME)
def dynamic_dual_level_mx_quant(input, *, impl="auto", **kwargs):
    """Dynamically quantize to dual-level MXFP4 on arch35.

    Scale groups are formed along the last input axis. Returns the packed
    values and both levels of scales. smooth_scale and round_mode are optional.
    Both auto and torch_npu use the installed TorchNPU implementation.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, **kwargs)
