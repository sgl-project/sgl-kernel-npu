"""Public sparse flash attention dispatch for the independent CANN operator."""

import math

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "attention.sparse_flash_attention", "cce", supported_generations={Generation.ARCH35}
)
def _cce_impl(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=2,
    return_softmax_lse=False,
):
    """Raw CCE kernel: fixed DSA prefill contract (arch35 only).

    Requires BSND layouts, a shared K/V latent (``key is value``) of 8192
    tokens, top-2048 sparse indices, heads in {16, 32, 64, 128} and
    ``scale_value=1/24``; batch size and query length are runtime-generic.
    Other configurations stay on the ascendc backend.
    Always returns the softmax max/sum alongside the attention output.
    """
    if layout_query != "BSND" or layout_kv != "BSND":
        raise ValueError("cce backend supports BSND query/KV layouts only")
    if query.dim() != 4 or query.size(0) < 1:
        raise ValueError("cce backend requires a [B,SQ,H,512] query")
    if query.size(2) not in (16, 32, 64, 128):
        raise ValueError("cce backend requires heads in {16, 32, 64, 128}")
    if key is not value:
        raise ValueError("cce backend requires the shared K/V latent (key is value)")
    if key.dim() != 4 or key.size(1) != 8192:
        raise ValueError("cce backend requires a [1,8192,1,512] K/V latent")
    if sparse_indices.shape[-1] != 2048:
        raise ValueError("cce backend requires top-2048 sparse indices")
    if not math.isclose(scale_value, 1.0 / 24.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("cce backend requires scale_value=1/24")
    if query_rope is None or key_rope is None:
        raise ValueError("cce backend requires query_rope and key_rope")
    return torch.ops.cce.sparse_flash_attention(
        query, query_rope, key, key_rope, sparse_indices
    )


@register_impl("attention.sparse_flash_attention", "auto")
@register_impl("attention.sparse_flash_attention", "ascendc")
def _ascendc_impl(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=2,
    return_softmax_lse=False,
):
    return torch.ops.npu.sgl_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
    )


@register_impl("attention.sparse_flash_attention", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.sparse_flash_attention import (
            sparse_flash_attention as torch_native_sparse_flash_attention,
        )

        _torch_native_fn = torch_native_sparse_flash_attention
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.sparse_flash_attention")
def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=2,
    return_softmax_lse=False,
    impl="auto",
):
    """Return (attention_out, softmax_max, softmax_sum); auto selects ascendc.

    ``sparse_mode=0`` with ``layout_kv='PA_BSND'``/``'TND'`` is the DSA path
    used by production. ``softmax_max``/``softmax_sum`` are empty unless
    ``return_softmax_lse`` is true.
    """
    return get_impl_by_name("attention.sparse_flash_attention", impl)(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
    )
