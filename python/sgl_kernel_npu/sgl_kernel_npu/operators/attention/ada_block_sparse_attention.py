"""Public AdaBlockSparseAttention dispatch for the independent attention CANN operator."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "attention.ada_block_sparse_attention",
    "auto",
    supported_generations={Generation.ARCH22},
)
@register_impl(
    "attention.ada_block_sparse_attention",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(
    query,
    key,
    value,
    sparse_mask,
    sparse_count_table,
    input_layout,
    sparse_size,
    num_heads,
    num_key_value_heads,
    scale_value,
    causal,
    inner_precise,
    pre_tokens,
    next_tokens,
    actual_seq_lengths,
    actual_seq_lengths_kv,
):
    import torch

    return torch.ops.npu.ada_block_sparse_attention(
        query,
        key,
        value,
        sparse_mask,
        sparse_count_table,
        input_layout,
        sparse_size,
        num_heads,
        num_key_value_heads,
        scale_value,
        causal,
        inner_precise,
        pre_tokens,
        next_tokens,
        actual_seq_lengths,
        actual_seq_lengths_kv,
    )


@register_impl("attention.ada_block_sparse_attention", "torch_native")
def _torch_native_impl(
    query,
    key,
    value,
    sparse_mask,
    sparse_count_table,
    input_layout,
    sparse_size,
    num_heads,
    num_key_value_heads,
    scale_value,
    causal,
    inner_precise,
    pre_tokens,
    next_tokens,
    actual_seq_lengths,
    actual_seq_lengths_kv,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.ada_block_sparse_attention import (
            ada_block_sparse_attention,
        )

        _torch_native_fn = ada_block_sparse_attention
    return _torch_native_fn(
        query,
        key,
        value,
        sparse_mask,
        sparse_count_table,
        input_layout=input_layout,
        sparse_size=sparse_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale_value,
        causal=causal,
        inner_precise=inner_precise,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
    )


@register_operator("attention.ada_block_sparse_attention")
def ada_block_sparse_attention(
    query,
    key,
    value,
    sparse_mask,
    sparse_count_table,
    *,
    input_layout="BNSD",
    sparse_size=128,
    num_heads=1,
    num_key_value_heads=1,
    scale_value=1.0,
    causal=True,
    inner_precise=1,
    pre_tokens=214748647,
    next_tokens=0,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    impl="auto",
):
    """Return block-sparse attention output for BNSD tensors.

    auto selects ascendc on arch22 without fallback. sparse_mask is int8
    [B, N, ceil(S1 / sparse_size), ceil(S2 / sparse_size)]; a non-zero entry
    enables the whole KV block for the matching query block.
    sparse_count_table is the per-row count of enabled blocks
    [B, N, ceil(S1 / sparse_size)].
    """
    if num_key_value_heads == 0:
        num_key_value_heads = num_heads
    return get_impl_by_name("attention.ada_block_sparse_attention", impl)(
        query,
        key,
        value,
        sparse_mask,
        sparse_count_table,
        input_layout,
        sparse_size,
        num_heads,
        num_key_value_heads,
        scale_value,
        causal,
        inner_precise,
        pre_tokens,
        next_tokens,
        actual_seq_lengths,
        actual_seq_lengths_kv,
    )
