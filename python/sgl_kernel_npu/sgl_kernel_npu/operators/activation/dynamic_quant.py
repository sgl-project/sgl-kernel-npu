"""Dynamic quantization dispatch."""

import torch_npu
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_IMPL_NAME = "activation.dynamic_quant"


@register_impl(_IMPL_NAME, "auto")
@register_impl(_IMPL_NAME, "torch_npu")
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_dynamic_quant(*args, **kwargs)


@register_impl(_IMPL_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    from sgl_kernel_npu.implementation.torch_native.activation.dynamic_quant import (
        dynamic_quant as reference,
    )

    return reference(*args, **kwargs)


@register_operator(_IMPL_NAME)
def dynamic_quant(*args, impl="auto", **kwargs):
    """Dispatch arguments and results to the selected dynamic quant operator."""
    return get_impl_by_name(_IMPL_NAME, impl)(*args, **kwargs)
