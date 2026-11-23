"""Packed C8 KV sparse attention dispatch for Ascend."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_OP_NAME = "attention.kv_quant_sparse_flash_attention"


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH22, Generation.ARCH35})
@register_impl(_OP_NAME, "ascendc", supported_generations={Generation.ARCH22, Generation.ARCH35})
def _ascendc_impl(*args, **kwargs):
    # Calls the PyTorch binding in csrc/attention/sparse_flash_attention/torch_adapter.cpp.
    return torch.ops.npu.kv_quant_sparse_flash_attention(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native", supported_generations={Generation.ARCH22, Generation.ARCH35})
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.kv_quant_sparse_flash_attention import (
            kv_quant_sparse_flash_attention as reference,
        )

        _torch_native_fn = reference
    return _torch_native_fn(*args, **kwargs)


@register_operator(_OP_NAME)
def kv_quant_sparse_flash_attention(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    scale_value: float,
    *,
    return_softmax_lse: bool = True,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run rank-local MLA sparse attention; auto selects ascendc.

    query is BF16 [T, H, 512]. query_rope is BF16 [T, H, 64] for
    656-byte KV rows, or [T, H, 0] for 528-byte rows. The latter store
    only the quantized latent followed by four FP32 scales at byte offset 512.
    Ascend C uses int8 on A2/A3 and uint8 carrying E4M3 bytes on A5.
    Both use [pages, page_size, 1, row_bytes].
    sparse_indices is int32 [T, 1, K] with trailing -1 padding.
    block_table is int32 [B, max_pages]. actual_seq_lengths_query and
    actual_seq_lengths_kv are int32 [B] query prefix sums and valid KV lengths.
    Tensors are on the same device. scale_value is the softmax score multiplier.
    impl selects auto, ascendc or the torch_native accuracy reference.
    Pass indices that already encode causal visibility for causal attention.
    Returns BF16 attention [T, H, 512] and FP32 softmax max/sum [1, T, H]
    or empty softmax tensors when return_softmax_lse is false.
    """
    return get_impl_by_name(_OP_NAME, impl)(
        query,
        query_rope,
        packed_kv,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        scale_value,
        return_softmax_lse=return_softmax_lse,
    )
