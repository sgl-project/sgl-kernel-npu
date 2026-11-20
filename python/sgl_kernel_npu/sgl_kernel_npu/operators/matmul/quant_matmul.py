"""Multiply quantized matrices with TorchNPU."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.quant_matmul"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-nn")
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_quant_matmul(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.matmul.torch_npu_quant_matmul import (
            quant_matmul,
        )

        _native_impl = quant_matmul
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def quant_matmul(x1, x2, scale, *, impl="auto", **kwargs):
    """Multiply quantized matrices with TorchNPU.

    x1 and x2 have compatible inner dimensions. scale describes x2. TorchNPU
    keywords include pertoken_scale, output_dtype, scale dtypes and group_sizes.
    Both auto and ops-nn call the installed TorchNPU operator.
    """
    return get_impl_by_name(_OP_NAME, impl)(x1, x2, scale, **kwargs)
