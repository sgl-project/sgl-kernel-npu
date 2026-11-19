"""Multiply quantized matrices with TorchNPU."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.quant_matmul"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_quant_matmul

        _torch_npu_fn = npu_quant_matmul
    return _torch_npu_fn(*args, **kwargs)


@register_operator(_OP_NAME)
def quant_matmul(x1, x2, scale, *, impl="auto", **kwargs):
    """Multiply quantized matrices with TorchNPU.

    x1 and x2 have compatible inner dimensions. scale describes x2. TorchNPU
    keywords include pertoken_scale, output_dtype, scale dtypes and group_sizes.
    Both auto and torch_npu use the installed TorchNPU implementation.
    """
    return get_impl_by_name(_OP_NAME, impl)(x1, x2, scale, **kwargs)
