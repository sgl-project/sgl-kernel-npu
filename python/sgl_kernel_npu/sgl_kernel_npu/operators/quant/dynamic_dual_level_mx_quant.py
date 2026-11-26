"""Dynamically quantize to dual-level MXFP4 on arch35."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "quant.dynamic_dual_level_mx_quant"
_native_impl = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "ops-nn", supported_generations={Generation.ARCH35})
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_dynamic_dual_level_mx_quant(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native", supported_generations={Generation.ARCH35})
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.quant.dynamic import (
            dynamic_dual_level_mx_quant,
        )

        _native_impl = dynamic_dual_level_mx_quant
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def dynamic_dual_level_mx_quant(input, *, impl="auto", **kwargs):
    """Dynamically quantize to dual-level MXFP4 on arch35.

    Scale groups are formed along the last input axis. Returns the packed
    values and both levels of scales. smooth_scale and round_mode are optional.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, **kwargs)
