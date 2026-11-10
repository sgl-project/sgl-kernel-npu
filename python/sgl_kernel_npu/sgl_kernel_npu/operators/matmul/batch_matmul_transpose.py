"""Public batch matmul with transposed batch/token axes and an output buffer."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "matmul.batch_matmul_transpose", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "matmul.batch_matmul_transpose",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(tensor_a, tensor_b, tensor_c, format_mode, quant_mode):
    return torch.ops.npu.batch_matmul_transpose(
        tensor_a, tensor_b, tensor_c, format_mode, quant_mode
    )


@register_impl("matmul.batch_matmul_transpose", "torch_native")
def _torch_native_impl(tensor_a, tensor_b, tensor_c, format_mode, quant_mode):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.matmul.batch_matmul_transpose import (
            batch_matmul_transpose,
        )

        _torch_native_fn = batch_matmul_transpose
    return _torch_native_fn(tensor_a, tensor_b, tensor_c, format_mode, quant_mode)


@register_operator("matmul.batch_matmul_transpose")
def batch_matmul_transpose(
    tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None, *, impl="auto"
):
    """Write einsum('mbk,bkn->mbn') into tensor_c and return None.

    auto selects ascendc for Ascend A2/A3 (arch22) without fallback.
    All tensors are contiguous and have the same float16/bfloat16 dtype and
    device. tensor_a is [M, B, K],
    tensor_c is [M, B, N], and 1 <= M <= 1024. tensor_b is [B, K, N] for ND
    (default), or explicitly packed [B, N/16, K, 16] for NZ. NZ requires K/N
    divisible by 16. quant_mode accepts the legacy mode names but does not
    enable quantization. Output storage must not overlap either input.
    """
    return get_impl_by_name("matmul.batch_matmul_transpose", impl)(
        tensor_a, tensor_b, tensor_c, format_mode, quant_mode
    )
