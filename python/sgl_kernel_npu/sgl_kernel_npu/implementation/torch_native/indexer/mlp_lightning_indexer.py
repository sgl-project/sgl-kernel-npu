"""Independent PyTorch reference for the LongCat2 MLP indexer."""

import torch
import torch.nn.functional as F


def _compute_scores(
    query,
    key,
    weights,
    cur_seq_lengths_query,
    cur_seq_lengths_key,
    block_table,
    init_tensor,
    local_tensor,
    layout_query,
    layout_key,
    kv_block_len,
    q_block_len,
    init_num,
    local_num,
    sparse_mode,
):
    """Return fp32 scores before output casting, including invalid -inf slots.

    Length tensors follow the kernel's B+1 prefix convention. A zero logical
    TND query advances one physical row, as in the upstream padding contract.
    Block scores average token scores before block-level causal masking; the
    last visible block is not forced into the selection.
    """
    if layout_query not in ("TND", "BSND") or layout_key not in (
        layout_query,
        "PA_BSND",
    ):
        raise NotImplementedError("The reference covers matching layouts or paged keys")
    if q_block_len != 1:
        raise NotImplementedError("The reference covers q_block_len=1")

    q_prefix = cur_seq_lengths_query.tolist()
    k_prefix = cur_seq_lengths_key.tolist()
    q_lengths = [end - start for start, end in zip(q_prefix, q_prefix[1:])]
    k_lengths = [end - start for start, end in zip(k_prefix, k_prefix[1:])]
    num_blocks = (max(k_lengths, default=0) + kv_block_len - 1) // kv_block_len
    rows = query.shape[0] * query.shape[1] if layout_query == "BSND" else query.shape[0]
    scores = torch.full(
        (rows, num_blocks),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    init_counts = init_tensor.tolist() if init_tensor is not None else None
    local_counts = local_tensor.tolist() if local_tensor is not None else None
    window_score = torch.finfo(torch.float32).max / 2
    q_offset = 0
    k_offset = 0

    for batch, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        if layout_query == "BSND":
            q_offset = batch * query.shape[1]
        if q_len == 0:
            q_offset += 1
            k_offset += max(k_len, 1)
            continue
        if k_len == 0:
            q_offset += q_len
            continue

        if layout_key == "PA_BSND":
            page_size = key.shape[1]
            page_count = (k_len + page_size - 1) // page_size
            pages = block_table[batch, :page_count].long()
            batch_key = key[pages].reshape(-1, key.shape[-1])[:k_len]
        elif layout_key == "TND":
            batch_key = key[k_offset : k_offset + k_len, 0]
        else:
            batch_key = key[batch, :k_len, 0]
        if layout_query == "TND":
            batch_query = query[q_offset : q_offset + q_len].float()
            batch_weights = weights[q_offset : q_offset + q_len].float()
        else:
            batch_query = query[batch, :q_len].float()
            batch_weights = weights[batch, :q_len].float()
        token_scores = (
            torch.einsum("qhd,kd->qhk", batch_query, batch_key.float()).relu()
            * batch_weights.unsqueeze(-1)
        ).sum(dim=1)

        blocks = (k_len + kv_block_len - 1) // kv_block_len
        token_scores = F.pad(token_scores, (0, blocks * kv_block_len - k_len))
        block_scores = token_scores.reshape(q_len, blocks, kv_block_len).mean(-1)
        block_scores = block_scores.clamp_max(window_score / 2)
        block_ids = torch.arange(blocks, device=query.device)
        visible = torch.full((q_len,), k_len, device=query.device, dtype=torch.int64)
        if sparse_mode == 3:
            visible = k_len - q_len + torch.arange(1, q_len + 1, device=query.device)
        valid = block_ids.unsqueeze(0) < (
            (visible + kv_block_len - 1) // kv_block_len
        ).unsqueeze(1)
        batch_init = init_counts[batch] if init_counts is not None else init_num
        batch_local = local_counts[batch] if local_counts is not None else local_num
        sink = block_ids < batch_init // kv_block_len
        # The native local window begins at the first whole block starting at
        # or after visible-local, including a partially visible final block.
        local = (block_ids * kv_block_len).unsqueeze(0) >= (
            visible - batch_local
        ).unsqueeze(1)
        if batch_local == 0:
            local = torch.zeros_like(local)
        block_scores = block_scores.masked_fill(sink.unsqueeze(0) | local, window_score)
        block_scores = block_scores.masked_fill(~valid, float("-inf"))
        scores[q_offset : q_offset + q_len, :blocks] = block_scores
        q_offset += q_len
        k_offset += k_len
    return scores


def mlp_lightning_indexer(
    query,
    key,
    weights,
    cur_seq_lengths_query=None,
    cur_seq_lengths_key=None,
    block_table=None,
    init_tensor=None,
    local_tensor=None,
    layout_query="BSND",
    layout_key="PA_BSND",
    sparse_count=2048,
    kv_block_len=1,
    q_block_len=1,
    init_num=0,
    local_num=0,
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    return_value=False,
):
    """Compute weighted ReLU scores, pool KV blocks and select sparse indices.

    pre_tokens and next_tokens are retained for signature compatibility; the
    native computation uses sparse_mode rather than these attributes. Disabled
    values retain the native tuple/shape contract and have no defined contents.
    """
    scores = _compute_scores(
        query,
        key,
        weights,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        block_table,
        init_tensor,
        local_tensor,
        layout_query,
        layout_key,
        kv_block_len,
        q_block_len,
        init_num,
        local_num,
        sparse_mode,
    )
    output_shape = (*query.shape[:-2], 1, sparse_count)
    flat_shape = (scores.shape[0], sparse_count)
    indices = torch.full(flat_shape, -1, dtype=torch.int32, device=query.device)
    values = torch.empty(flat_shape, dtype=query.dtype, device=query.device)
    if return_value:
        values.fill_(float("-inf"))
    count = min(sparse_count, scores.shape[-1])
    if count:
        selected_values, selected_indices = scores.topk(count, dim=-1)
        selected_indices = selected_indices.masked_fill(
            selected_values.isneginf(), -1
        ).to(torch.int32)
        indices[:, :count] = selected_indices
        if return_value:
            # Both fp16 and bf16 native paths cap returned scores at 65504.
            values[:, :count] = selected_values.clamp_max(65504).to(query.dtype)
    return indices.reshape(output_shape), values.reshape(output_shape)
