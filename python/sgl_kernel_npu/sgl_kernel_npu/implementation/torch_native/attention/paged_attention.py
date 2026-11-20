"""CPU PyTorch reference for legacy ATB attention with ND paged caches."""

import torch

from .fused_infer_attention_score import dense_attention
from .sparse_flash_attention import _paged_sequence


def _run_paged_attention(
    query,
    key_cache,
    value_cache,
    block_table,
    query_lengths,
    context_lens,
    scale_value,
    num_heads,
    num_kv_heads,
    out,
    *,
    mask=None,
    value_width=None,
):
    key = key_cache.detach().cpu()
    value = value_cache.detach().cpu()
    if key.ndim != 4 or value.ndim != 4 or key.size(2) != num_kv_heads:
        raise NotImplementedError("torch_native requires ND paged caches")
    if value_width is not None:
        value = value[..., :value_width]
    q = query.detach().cpu().reshape(-1, num_heads, key.size(-1))
    rows = block_table.detach().cpu()
    if len(query_lengths) != len(context_lens) or rows.size(0) != len(context_lens):
        raise ValueError("Expected one block-table row and lengths per request")
    if sum(query_lengths) != q.size(0):
        raise ValueError("Query lengths must cover all query tokens")
    result = q.new_zeros((q.size(0), num_heads, value.size(-1)))
    start = 0
    for i, (q_len, kv_len) in enumerate(zip(query_lengths, context_lens)):
        end = start + q_len
        keys = _paged_sequence(key, rows[i], kv_len)
        values = _paged_sequence(value, rows[i], kv_len)
        if q_len and kv_len:
            result[start:end] = (
                dense_attention(
                    q[start:end].transpose(0, 1).unsqueeze(0),
                    keys.transpose(0, 1).unsqueeze(0),
                    values.transpose(0, 1).unsqueeze(0),
                    scale_value,
                    mask=mask[:q_len, :kv_len] if mask is not None else None,
                )[0]
                .squeeze(0)
                .transpose(0, 1)
            )
        start = end
    out.copy_(result.reshape(out.shape).to(out.device))


def flash_attention_qlens(
    query,
    key_cache,
    value_cache,
    mask,
    block_table,
    seq_len,
    context_lens,
    scale_value,
    num_heads,
    num_kv_heads,
    out,
):
    if mask is not None and mask.ndim != 2:
        raise NotImplementedError("torch_native requires a 2D prefill mask")
    _run_paged_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        list(seq_len),
        list(context_lens),
        scale_value,
        num_heads,
        num_kv_heads,
        out,
        mask=mask.detach().cpu() if mask is not None else None,
    )


def paged_attention(
    query,
    key_cache,
    value_cache,
    num_heads,
    num_kv_heads,
    scale_value,
    block_table,
    context_lens,
    out,
    *,
    workspace=None,
):
    _run_paged_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        [1] * len(context_lens),
        list(context_lens),
        scale_value,
        num_heads,
        num_kv_heads,
        out,
    )


def paged_attention_mla(
    query,
    key_cache,
    num_kv_heads,
    num_heads,
    scale_value,
    block_table,
    context_lens,
    mla_vheadsize,
    out,
    *,
    workspace=None,
):
    _run_paged_attention(
        query,
        key_cache,
        key_cache,
        block_table,
        [1] * len(context_lens),
        list(context_lens),
        scale_value,
        num_heads,
        num_kv_heads,
        out,
        value_width=mla_vheadsize,
    )
