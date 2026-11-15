"""Same-shape addition with explicit AICPU or PyTorch implementation selection."""

import torch
import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("examples.add_example", "auto")
@register_impl("examples.add_example", "aicpu")
def _aicpu_impl(x1, x2):
    return torch.ops.npu.sgl_add_example(x1, x2)


@register_impl("examples.add_example", "torch_native")
def _torch_native_impl(x1, x2):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.examples.add_example import add_example

        _torch_native_fn = add_example

    return _torch_native_fn(x1, x2)


@register_operator("examples.add_example")
def add_example(x1, x2, *, impl="auto"):
    """Return a new tensor containing x1 + x2, without modifying either input.

    Inputs must be nonempty, same-shape, contiguous ND float32 or int32 tensors
    with the same dtype on the same NPU. Broadcasting is rejected.
    auto selects aicpu without fallback; torch_native is an independent reference.
    """
    impl_fn = get_impl_by_name("examples.add_example", impl)
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        raise TypeError("inputs must be torch tensors")
    if x1.device.type != "npu" or x2.device != x1.device:
        raise ValueError("inputs must be on the same NPU")
    if x1.shape != x2.shape:
        raise ValueError(
            "inputs must have identical shapes; broadcasting is not supported"
        )
    if x1.dtype != x2.dtype:
        raise ValueError("inputs must have the same dtype")
    if x1.dtype not in (torch.float32, torch.int32):
        raise ValueError("inputs must be float32 or int32")
    if not x1.is_contiguous() or not x2.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if x1.numel() == 0:
        raise ValueError("empty inputs are not supported")
    # ACL_FORMAT_ND is 2; do not silently convert a blocked storage layout.
    if torch_npu.get_npu_format(x1) != 2 or torch_npu.get_npu_format(x2) != 2:
        raise ValueError("inputs must use ND storage format")
    return impl_fn(x1, x2)
