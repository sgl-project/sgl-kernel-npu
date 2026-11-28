"""Public grouped matmul, SwiGLU and dynamic quantization operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl(
    "moe.grouped_matmul_swiglu_quant", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "moe.grouped_matmul_swiglu_quant",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_impl(x, weight, weight_scale, x_scale, group_list, swiglu_limit):
    global _cann_fn
    if _cann_fn is None:
        from ..._third_party.sgl_ops_transformer_gmm.ops.gmm.sgl_grouped_matmul_swiglu_quant import (
            SglGroupedMatmulSwigluQuantOpBuilder,
        )

        _cann_fn = (
            SglGroupedMatmulSwigluQuantOpBuilder().load().grouped_matmul_swiglu_quant
        )
    return _cann_fn(x, weight, weight_scale, x_scale, group_list, swiglu_limit)


@register_impl("moe.grouped_matmul_swiglu_quant", "torch_native")
def _torch_native_impl(x, weight, weight_scale, x_scale, group_list, swiglu_limit):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.moe.grouped_matmul_swiglu_quant import (
            grouped_matmul_swiglu_quant,
        )

        _torch_native_fn = grouped_matmul_swiglu_quant
    return _torch_native_fn(x, weight, weight_scale, x_scale, group_list, swiglu_limit)


@register_operator("moe.grouped_matmul_swiglu_quant")
def grouped_matmul_swiglu_quant(
    x, weight, weight_scale, x_scale, group_list, swiglu_limit=0.0, *, impl="auto"
):
    """Return INT8 [R,N/2] and FP32 [R] scales for INT8 NZ [E,K,N] weights.

    group_list contains cumulative INT64 expert ends. Only rows before its last
    entry are valid. swiglu_limit=0 disables clipping. W4A8 is unsupported.
    """
    if x.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("Fused GMM requires INT8 activations and weights")
    return get_impl_by_name("moe.grouped_matmul_swiglu_quant", impl)(
        x, weight, weight_scale, x_scale, group_list, swiglu_limit
    )
