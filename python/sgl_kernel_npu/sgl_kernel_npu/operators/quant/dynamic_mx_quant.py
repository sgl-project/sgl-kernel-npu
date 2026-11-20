"""Dynamically quantize an input axis to MX format on arch35."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "quant.dynamic_mx_quant"
_native_impl = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "ops-nn", supported_generations={Generation.ARCH35})
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_dynamic_mx_quant(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native", supported_generations={Generation.ARCH35})
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.quant.dynamic import dynamic_mx_quant

        _native_impl = dynamic_mx_quant
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def dynamic_mx_quant(input, *, impl="auto", **kwargs):
    """Dynamically quantize an input axis to MX format on arch35.

    The floating-point input and TorchNPU options such as axis, dst_type and
    block_size determine the packed values and E8M0 scales returned.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, **kwargs)
