"""Multiply quantized 3D matrices with configurable permutations on arch35."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.transpose_quant_batchmatmul"
_native_impl = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "ops-nn", supported_generations={Generation.ARCH35})
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_transpose_quant_batchmatmul(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native", supported_generations={Generation.ARCH35})
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.matmul.torch_npu_quant_matmul import (
            transpose_quant_batchmatmul,
        )

        _native_impl = transpose_quant_batchmatmul
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def transpose_quant_batchmatmul(x1, x2, dtype, *, impl="auto", **kwargs):
    """Multiply quantized 3D matrices with configurable permutations on arch35.

    x1 has shape [M, B, K]. Use the TorchNPU perm_x1, perm_x2 and perm_y
    keywords to specify batch and matrix axes. dtype is the output dtype.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(x1, x2, dtype, **kwargs)
