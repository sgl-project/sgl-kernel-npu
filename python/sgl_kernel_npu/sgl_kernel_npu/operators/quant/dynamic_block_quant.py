"""Dynamically quantize blocks and return the quantized tensor and scales."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "quant.dynamic_block_quant"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_dynamic_block_quant

        _torch_npu_fn = npu_dynamic_block_quant
    return _torch_npu_fn(*args, **kwargs)


@register_operator(_OP_NAME)
def dynamic_block_quant(x, *, impl="auto", **kwargs):
    """Dynamically quantize blocks and return the quantized tensor and scales.

    x is a 2D or 3D floating-point tensor. TorchNPU accepts block sizes,
    dst_type and rounding options. The FP8 path used by DSA requires arch35.
    Both auto and torch_npu use the installed TorchNPU implementation.
    """
    return get_impl_by_name(_OP_NAME, impl)(x, **kwargs)
