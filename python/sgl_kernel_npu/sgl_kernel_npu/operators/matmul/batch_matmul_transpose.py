"""Public batch matmul with transposed batch/token axes."""

import torch
import torch_npu

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.batch_matmul_transpose"
_FUSED_TRANSPOSE_LIMIT = 65536
_torch_native_fn = None


@register_impl(_OP_NAME, "ascendc", supported_generations={Generation.ARCH22})
def _ascendc_impl(tensor_a, tensor_b, *, is_dsa=False):
    if not tensor_b.is_contiguous():
        raise ValueError("tensor_b must be contiguous")

    m, batch, _ = tensor_a.shape
    _, _, n = tensor_b.shape
    output = tensor_a.new_empty((m, batch, n))
    torch.ops.npu.batch_matmul_transpose(tensor_a.contiguous(), tensor_b, output)
    return output


@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(tensor_a, tensor_b, *, is_dsa=False):
    perm_x1 = (1, 0, 2)
    if (
        is_dsa
        and get_hardware().generation is Generation.ARCH22
        and tensor_a.shape[1] * tensor_a.shape[2] >= _FUSED_TRANSPOSE_LIMIT
    ):
        # Bypass the A2/A3 fused input-transpose stride limit for DSA.
        tensor_a = tensor_a.transpose(0, 1).contiguous()
        perm_x1 = (0, 1, 2)

    return torch_npu.npu_transpose_batchmatmul(
        tensor_a,
        tensor_b,
        perm_x1=perm_x1,
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )


@register_impl(_OP_NAME, "auto")
def _auto_impl(tensor_a, tensor_b, *, is_dsa=False):
    if is_dsa:
        return _torch_npu_impl(tensor_a, tensor_b, is_dsa=True)
    if get_hardware().generation is not Generation.ARCH22:
        return _torch_npu_impl(tensor_a, tensor_b)

    m, batch, k = tensor_a.shape
    _, _, n = tensor_b.shape
    # CANN 9.1 fused transpose matmul guard for MLA on A2/A3.
    if max(m, batch * k, n) >= _FUSED_TRANSPOSE_LIMIT:
        return _ascendc_impl(tensor_a, tensor_b)
    return _torch_npu_impl(tensor_a, tensor_b)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(tensor_a, tensor_b, *, is_dsa=False):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.matmul.batch_matmul_transpose import (
            batch_matmul_transpose,
        )

        _torch_native_fn = batch_matmul_transpose
    return _torch_native_fn(tensor_a, tensor_b)


@register_operator(_OP_NAME)
def batch_matmul_transpose(tensor_a, tensor_b, *, is_dsa=False, impl="auto"):
    """Return einsum('mbk,bkn->mbn') for ND inputs [M, B, K] and [B, K, N].

    With is_dsa=True, auto selects torch_npu. On arch22, the torch_npu backend
    pre-transposes DSA inputs to [B, M, K] when B * K >= 65536. For MLA,
    auto selects ascendc on arch22 when M, B * K or N reaches 65536, and
    torch_npu otherwise. Explicit impl selection bypasses auto dispatch.

    ascendc and torch_native require matching FP16/BF16 dtype and device,
    contiguous weights, and 1 <= M <= 1024. torch_npu follows the shape and
    dtype limits of the installed TorchNPU and CANN versions.
    """
    return get_impl_by_name(_OP_NAME, impl)(tensor_a, tensor_b, is_dsa=is_dsa)
