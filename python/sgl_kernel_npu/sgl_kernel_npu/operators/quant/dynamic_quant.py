"""Dynamically quantize floating-point tokens with TorchNPU."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "quant.dynamic_quant"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-nn")
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_dynamic_quant(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.quant.dynamic import dynamic_quant

        _native_impl = dynamic_quant
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def dynamic_quant(input, *, impl="auto", **kwargs):
    """Dynamically quantize floating-point tokens with TorchNPU.

    Returns quantized values and scales. Pass dst_type and optional smoothing
    or MoE group metadata using the TorchNPU keyword arguments.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, **kwargs)
