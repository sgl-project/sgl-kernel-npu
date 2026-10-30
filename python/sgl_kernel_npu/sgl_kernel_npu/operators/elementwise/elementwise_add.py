"""Public element-wise add with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("elementwise.elementwise_add", "auto")
@register_impl("elementwise.elementwise_add", "ascendc")
def _ascendc_impl(x, y):
    import torch

    return torch.ops.npu.elementwise_add(x, y)


@register_impl("elementwise.elementwise_add", "torch_native")
def _torch_native_impl(x, y):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.elementwise.elementwise_add import (
            elementwise_add,
        )

        _torch_native_fn = elementwise_add

    return _torch_native_fn(x, y)


@register_operator("elementwise.elementwise_add")
def elementwise_add(x, y, *, impl="auto"):
    """Add two tensors element-wise, returning a new tensor; auto selects ascendc.

    x and y must be contiguous NPU tensors with the same shape and dtype
    (float16 or float32). Any rank is accepted and the result keeps x's shape.
    auto is an alias of the Ascend C entry point, not a fallback.
    """
    impl_fn = get_impl_by_name("elementwise.elementwise_add", impl)
    return impl_fn(x, y)
