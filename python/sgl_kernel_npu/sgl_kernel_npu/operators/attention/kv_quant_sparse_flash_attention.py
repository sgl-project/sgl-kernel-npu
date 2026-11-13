"""Packed FP8 KV sparse attention dispatch for Ascend 950."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_OP_NAME = "attention.kv_quant_sparse_flash_attention"


@register_impl(_OP_NAME, "auto", supported_generations={Generation.ARCH35})
@register_impl(_OP_NAME, "ascendc", supported_generations={Generation.ARCH35})
def _ascendc_impl(*args):
    return torch.ops.npu.sgl_kv_quant_sparse_flash_attention(*args)


@register_impl(_OP_NAME, "torch_native", supported_generations={Generation.ARCH35})
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.kv_quant_sparse_flash_attention import (
            kv_quant_sparse_flash_attention as reference,
        )

        _torch_native_fn = reference
    return _torch_native_fn(*args)


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
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run rank-local MLA sparse attention; auto selects ascendc on arch35.

    query and query_rope are BF16 [T, H, 512] and [T, H, 64]. packed_kv
    is uint8 [pages, page_size, 1, 656] with FP8 latent, BF16 RoPE and
    FP32 scales. sparse_indices is int32 [T, 1, K] with trailing -1 padding.
    block_table is int32 [B, max_pages]. actual_seq_lengths_query and
    actual_seq_lengths_kv are int32 [B] query prefix sums and valid KV lengths.
    Tensors are on the same device. scale_value is the softmax score multiplier.
    impl selects auto, ascendc or the torch_native accuracy reference.
    Returns BF16 attention [T, H, 512] and FP32 softmax max/sum [1, T, H].
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
    )
