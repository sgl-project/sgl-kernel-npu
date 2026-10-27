"""Public l1_norm with implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.l1_norm", "auto")
@register_impl("norm.l1_norm", "triton")
def _triton_impl(input):
    from ...implementation.triton.norm.l1_norm import l1_norm_kernel
    from ...utils.triton_utils import get_device_properties

    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    output = torch.empty(
        batch_size, hidden_size, device=input.device, dtype=torch.float32
    )
    _, num_vectorcore = get_device_properties()
    l1_norm_kernel[(num_vectorcore,)](
        input, output, batch_size, hidden_size, num_vectorcore
    )
    return output


@register_impl("norm.l1_norm", "torch_native")
def _torch_native_impl(input):
    from ...implementation.torch_native.norm.l1_norm import l1_norm

    return l1_norm(input)


@register_operator("norm.l1_norm")
def l1_norm(input, *, impl="auto"):
    """Use the original Triton implementation by default, or the PyTorch reference."""
    return get_impl_by_name("norm.l1_norm", impl)(input)
