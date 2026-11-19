"""Multiply floating-point activations by quantized weights."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.weight_quant_batchmatmul"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_weight_quant_batchmatmul

        _torch_npu_fn = npu_weight_quant_batchmatmul
    return _torch_npu_fn(*args, **kwargs)


@register_operator(_OP_NAME)
def weight_quant_batchmatmul(x, weight, antiquant_scale, *args, impl="auto", **kwargs):
    """Multiply floating-point activations by quantized weights.

    x ends in K. weight stores the K-by-N matrix, which can be packed.
    antiquant_scale supplies the per-channel or group-wise weight scales.
    TorchNPU optional arguments include antiquant_offset, bias and antiquant_group_size.
    Both auto and torch_npu use the installed TorchNPU implementation.
    """
    return get_impl_by_name(_OP_NAME, impl)(x, weight, antiquant_scale, *args, **kwargs)
