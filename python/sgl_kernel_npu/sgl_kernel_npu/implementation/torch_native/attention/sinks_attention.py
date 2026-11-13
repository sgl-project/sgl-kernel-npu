import torch


def _attention(
    query,
    k_cache,
    v_cache,
    sinks,
    seq_lens,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    dim = query.shape[-1] // q_head_num
    page_size = k_cache.shape[1]
    out = torch.empty_like(query)
    begin = 0
    for request, length in enumerate(seq_lens):
        length = int(length)
        context = int(context_lens[request])
        pages = block_tables[request, : (context + page_size - 1) // page_size].long()
        keys = k_cache[pages].reshape(-1, k_head_num, dim)[:context].float()
        values = v_cache[pages].reshape(-1, k_head_num, dim)[:context].float()
        keys = keys.repeat_interleave(q_head_num // k_head_num, 1)
        values = values.repeat_interleave(q_head_num // k_head_num, 1)
        for position in range(length):
            end = context - length + position + 1
            start = (
                max(0, end - sliding_window_size) if sliding_window_size != -1 else 0
            )
            q = query[begin + position].view(q_head_num, dim).float()
            score = torch.einsum("hd,thd->ht", q, keys[start:end]) * scale
            scores = torch.cat((score, sinks.float().view(-1, 1)), -1)
            prob = scores.softmax(-1)[:, :-1]
            out[begin + position] = (
                torch.einsum("ht,thd->hd", prob, values[start:end])
                .flatten()
                .to(query.dtype)
            )
        begin += length
    return out


def attention_sinks(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    return _attention(
        query,
        k_cache,
        v_cache,
        sinks,
        [1] * query.shape[0],
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
    )


def attention_sinks_prefill(
    query,
    k_cache,
    v_cache,
    sinks,
    seq_lens,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    return _attention(
        query,
        k_cache,
        v_cache,
        sinks,
        seq_lens,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
    )
