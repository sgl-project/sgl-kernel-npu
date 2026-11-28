# SPDX-License-Identifier: Apache-2.0
"""Independent PyTorch reference for paged attention with sink logits.

The kernels under test are never called. Each request gathers its K/V pages
through ``block_tables``, expands shared KV heads, and evaluates the softmax in
FP32. The sink is appended as one extra logit whose value column is dropped,
which is exactly a zero-value contribution to the output.
"""

import torch

# Query rows per score chunk: bounds the [rows, heads, kv] score tensor.
_REFERENCE_QUERY_CHUNK = 128


def _query_to_kv_heads(num_q_heads, num_kv_heads, gqa_interleave, device):
    heads = torch.arange(num_q_heads, device=device)
    if gqa_interleave:
        return heads % num_kv_heads
    return heads // (num_q_heads // num_kv_heads)


def _gather_pages(cache, block_tables, batch_id, kv_len, page_size):
    """Materialise ``[kv_len, Hkv, D]`` valid KV rows for one request."""
    num_pages = (kv_len + page_size - 1) // page_size
    if num_pages > block_tables.shape[1]:
        raise ValueError(
            "block_tables has {} pages but a KV length of {} needs {}".format(
                block_tables.shape[1], kv_len, num_pages
            )
        )
    pages = block_tables[batch_id, :num_pages].long()
    return cache[pages].reshape(-1, cache.shape[2], cache.shape[3])[:kv_len]


def _attend(query, key, value, head_map, sinks, scale, visible):
    """Attention for ``[rows, Hq, D]`` queries and ``[kv, Hkv, D]`` K/V.

    ``visible`` is ``[rows, kv]``; every row must expose at least one key.
    """
    key = key.index_select(1, head_map)
    value = value.index_select(1, head_map)
    scores = torch.einsum("qhd,khd->hqk", query, key) * scale
    scores = scores.masked_fill(~visible[None, :, :], float("-inf"))
    if sinks is None:
        probs = torch.softmax(scores, dim=-1)
    else:
        sink_column = sinks.view(-1, 1, 1).expand(-1, scores.shape[1], -1)
        extended = torch.cat((scores, sink_column), dim=-1)
        probs = torch.softmax(extended, dim=-1)[..., :-1]
    return torch.einsum("hqk,khd->qhd", probs, value)


def paged_attention_sink_prefill(
    q,
    k_cache,
    v_cache,
    cu_q_lens,
    seqlens_kv,
    block_tables,
    sinks,
    gqa_interleave,
    softmax_scale,
):
    """Reference implementation of causal paged prefill with sink logits."""
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    head_map = _query_to_kv_heads(num_q_heads, num_kv_heads, gqa_interleave, q.device)
    scale = softmax_scale
    sink_logits = None if sinks is None else sinks.float()

    boundaries = [int(value) for value in cu_q_lens]
    output = torch.zeros_like(q)
    for batch_id in range(len(boundaries) - 1):
        start, end = boundaries[batch_id], boundaries[batch_id + 1]
        query_len = end - start
        if query_len <= 0:
            continue

        kv_len = query_len if seqlens_kv is None else int(seqlens_kv[batch_id])
        if kv_len < query_len:
            raise ValueError(
                "seqlens_kv[{}] ({}) must be >= its query length ({})".format(
                    batch_id, kv_len, query_len
                )
            )
        kv_cache_len = kv_len - query_len

        key = _gather_pages(k_cache, block_tables, batch_id, kv_len, page_size).float()
        value = _gather_pages(
            v_cache, block_tables, batch_id, kv_len, page_size
        ).float()
        # New queries sit after the cached prefix, so query i may read keys
        # through position kv_cache_len + i.
        key_positions = torch.arange(kv_len, device=q.device)
        query_positions = kv_cache_len + torch.arange(query_len, device=q.device)
        visible = key_positions[None, :] <= query_positions[:, None]

        for chunk_start in range(0, query_len, _REFERENCE_QUERY_CHUNK):
            chunk_end = min(chunk_start + _REFERENCE_QUERY_CHUNK, query_len)
            row_start, row_end = start + chunk_start, start + chunk_end
            output[row_start:row_end] = _attend(
                q[row_start:row_end].float(),
                key,
                value,
                head_map,
                sink_logits,
                scale,
                visible[chunk_start:chunk_end],
            ).to(q.dtype)
    return output


def paged_attention_sink_decode(
    q,
    k_cache,
    v_cache,
    seqlens,
    block_tables,
    sinks,
    gqa_interleave,
    softmax_scale,
    max_kv_len=None,
):
    """Reference implementation of paged decode with sink logits."""
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    head_map = _query_to_kv_heads(num_q_heads, num_kv_heads, gqa_interleave, q.device)
    sink_logits = None if sinks is None else sinks.float()

    output = torch.zeros_like(q)
    for batch_id in range(q.shape[0]):
        kv_len = int(seqlens[batch_id])
        if kv_len <= 0:
            continue
        key = _gather_pages(k_cache, block_tables, batch_id, kv_len, page_size).float()
        value = _gather_pages(
            v_cache, block_tables, batch_id, kv_len, page_size
        ).float()
        # The current token is the last KV entry, so the whole KV row is visible.
        visible = torch.ones((1, kv_len), dtype=torch.bool, device=q.device)
        output[batch_id] = _attend(
            q[batch_id : batch_id + 1].float(),
            key,
            value,
            head_map,
            sink_logits,
            softmax_scale,
            visible,
        )[0].to(q.dtype)
    return output
