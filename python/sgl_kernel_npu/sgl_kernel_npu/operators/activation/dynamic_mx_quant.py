"""Dynamic MX quantization dispatch."""

import torch_npu
from sgl_kernel_npu.operators.utils.generation import Generation
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_IMPL_NAME = "activation.dynamic_mx_quant"
_ONLY_A5 = {Generation.ARCH35}


@register_impl(_IMPL_NAME, "auto", supported_generations=_ONLY_A5)
@register_impl(_IMPL_NAME, "torch_npu", supported_generations=_ONLY_A5)
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_dynamic_mx_quant(*args, **kwargs)


@register_impl(_IMPL_NAME, "torch_native", supported_generations=_ONLY_A5)
def _torch_native_impl(*args, **kwargs):
    from sgl_kernel_npu.implementation.torch_native.activation.dynamic_mx_quant import (
        dynamic_mx_quant as reference,
    )

    return reference(*args, **kwargs)


@register_operator(_IMPL_NAME)
def dynamic_mx_quant(*args, impl="auto", **kwargs):
    """Dispatch arguments and results to the selected dynamic MX quant operator."""
    return get_impl_by_name(_IMPL_NAME, impl)(*args, **kwargs)
