"""Public ELU operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("activation.elu", "auto")
@register_impl("activation.elu", "ascendc")
def _ascendc_impl(x, alpha):
    import torch

    return torch.ops.npu.elu(x, alpha)


@register_impl("activation.elu", "torch_native")
def _torch_native_impl(x, alpha):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.elu import elu

        _torch_native_fn = elu
    return _torch_native_fn(x, alpha)


@register_operator("activation.elu")
def elu(x, alpha=1.0, *, impl="auto"):
    """Element-wise ELU, returning a new tensor. ``auto`` selects Ascend C.

    ``x`` is a contiguous float16/float32 tensor on the NPU with at least one
    dimension; any shape is accepted because the kernel works on the flattened
    element sequence. ``alpha`` is a finite float coefficient, defaulting to
    ``1.0``. The output keeps ``x``'s shape, dtype and device.
    """
    impl_fn = get_impl_by_name("activation.elu", impl)
    return impl_fn(x, alpha)
