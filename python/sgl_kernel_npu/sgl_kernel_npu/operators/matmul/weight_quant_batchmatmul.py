"""Multiply floating-point activations by quantized weights."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.weight_quant_batchmatmul"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-nn")
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_weight_quant_batchmatmul(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.matmul.torch_npu_quant_matmul import (
            weight_quant_batchmatmul,
        )

        _native_impl = weight_quant_batchmatmul
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def weight_quant_batchmatmul(x, weight, antiquant_scale, *args, impl="auto", **kwargs):
    """Multiply floating-point activations by quantized weights.

    x ends in K. weight stores the K-by-N matrix, which can be packed.
    antiquant_scale supplies the per-channel or group-wise weight scales.
    TorchNPU optional arguments include antiquant_offset, bias and antiquant_group_size.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(x, weight, antiquant_scale, *args, **kwargs)
