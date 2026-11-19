"""Multiply quantized 3D matrices with configurable permutations on arch35."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.transpose_quant_batchmatmul"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "torch_npu", supported_generations={Generation.ARCH35})
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_transpose_quant_batchmatmul

        _torch_npu_fn = npu_transpose_quant_batchmatmul
    return _torch_npu_fn(*args, **kwargs)


@register_operator(_OP_NAME)
def transpose_quant_batchmatmul(x1, x2, dtype, *, impl="auto", **kwargs):
    """Multiply quantized 3D matrices with configurable permutations on arch35.

    x1 has shape [M, B, K]. Use the TorchNPU perm_x1, perm_x2 and perm_y
    keywords to specify batch and matrix axes. dtype is the output dtype.
    Both auto and torch_npu use the installed TorchNPU implementation.
    """
    return get_impl_by_name(_OP_NAME, impl)(x1, x2, dtype, **kwargs)
