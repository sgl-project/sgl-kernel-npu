"""Dynamically quantize blocks and return the quantized tensor and scales."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "quant.dynamic_block_quant"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-nn")
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_dynamic_block_quant(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.quant.dynamic import dynamic_block_quant

        _native_impl = dynamic_block_quant
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def dynamic_block_quant(x, *, impl="auto", **kwargs):
    """Dynamically quantize blocks and return the quantized tensor and scales.

    x is a 2D or 3D floating-point tensor. TorchNPU accepts block sizes,
    dst_type and rounding options. The FP8 path used by DSA requires arch35.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(x, **kwargs)
