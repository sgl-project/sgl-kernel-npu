"""Public batch matmul with transposed batch/token axes."""

from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "matmul.batch_matmul_transpose"
_torch_native_fn = None


@register_impl(_OP_NAME, "ascendc", supported_generations={Generation.ARCH22})
def _ascendc_impl(tensor_a, tensor_b, *, is_prefill=None):
    import torch

    m, batch, _ = tensor_a.shape
    _, _, n = tensor_b.shape
    output = tensor_a.new_empty((m, batch, n))
    torch.ops.npu.batch_matmul_transpose(tensor_a.contiguous(), tensor_b, output)
    return output


@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(tensor_a, tensor_b, *, is_prefill=None):
    import torch_npu

    return torch_npu.npu_transpose_batchmatmul(
        tensor_a,
        tensor_b,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )


@register_impl(_OP_NAME, "auto")
def _auto_impl(tensor_a, tensor_b, *, is_prefill=None):
    if get_hardware().generation is not Generation.ARCH22:
        return _torch_npu_impl(tensor_a, tensor_b)

    if is_prefill is None:
        m, batch, k = tensor_a.shape
        _, _, n = tensor_b.shape
        # CANN 9.1 fused transpose matmul guard for MLA on A2/A3.
        use_ascendc = max(m, batch * k, n) >= 65536
    else:
        use_ascendc = not is_prefill

    impl = _ascendc_impl if use_ascendc else _torch_npu_impl
    return impl(tensor_a, tensor_b)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(tensor_a, tensor_b, *, is_prefill=None):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.matmul.batch_matmul_transpose import (
            batch_matmul_transpose,
        )

        _torch_native_fn = batch_matmul_transpose
    return _torch_native_fn(tensor_a, tensor_b)


@register_operator(_OP_NAME)
def batch_matmul_transpose(tensor_a, tensor_b, *, is_prefill=None, impl="auto"):
    """Return einsum('mbk,bkn->mbn') for ND inputs [M, B, K] and [B, K, N].

    On arch22, auto uses the MLA shape guard when is_prefill is None. For DSA,
    True selects torch_npu and False selects ascendc. Other registered
    generations use torch_npu. Explicit impl selection bypasses this policy.

    ascendc and torch_native require matching FP16/BF16 dtype and device,
    contiguous weights, and 1 <= M <= 1024. torch_npu follows the shape and
    dtype limits of the installed TorchNPU and CANN versions.
    """
    return get_impl_by_name(_OP_NAME, impl)(tensor_a, tensor_b, is_prefill=is_prefill)
