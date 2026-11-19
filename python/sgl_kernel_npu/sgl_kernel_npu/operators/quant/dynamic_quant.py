"""Dynamically quantize floating-point tokens with TorchNPU."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "quant.dynamic_quant"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_dynamic_quant

        _torch_npu_fn = npu_dynamic_quant
    return _torch_npu_fn(*args, **kwargs)


@register_operator(_OP_NAME)
def dynamic_quant(input, *, impl="auto", **kwargs):
    """Dynamically quantize floating-point tokens with TorchNPU.

    Returns quantized values and scales. Pass dst_type and optional smoothing
    or MoE group metadata using the TorchNPU keyword arguments.
    Both auto and torch_npu use the installed TorchNPU implementation.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, **kwargs)
