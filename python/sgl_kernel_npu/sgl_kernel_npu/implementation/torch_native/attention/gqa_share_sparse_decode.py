"""Independent sparse GQA decode attention reference."""

import torch


def gqa_share_sparse_decode(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
):
    batch, qheads, dim = q.shape
    kvheads = k_cache_bnsd.shape[2]
    group = qheads // kvheads
    if topk_idx.shape[0] == qheads:
        topk_idx = topk_idx.reshape(kvheads, group, batch, -1)[:, 0]
    elif topk_idx.shape[0] != kvheads:
        raise ValueError("topk heads must match query or key heads")
    scale = dim**-0.5 if sm_scale is None else sm_scale
    out = torch.zeros_like(q)
    for b in range(batch):
        length = int(seq_lens[b].item())
        for h in range(qheads):
            kh = h // group
            keys, values = [], []
            for logical in topk_idx[kh, b].tolist():
                if logical < 0:
                    continue
                count = min(block_size, max(0, length - logical * block_size))
                if count == 0:
                    continue
                if req_to_token is None:
                    page = int(block_table[b, logical].item())
                else:
                    request = int(req_pool_indices[b].item())
                    col = min(logical * block_size, req_to_token.shape[1] - 1)
                    page = int(req_to_token[request, col].item()) // block_size
                    if sanitize_page_ids:
                        page = min(max(page, 0), num_pages - 1)
                keys.append(k_cache_bnsd[page, :count, kh].float())
                values.append(v_cache_bnsd[page, :count, kh].float())
            if not keys and sink is None:
                continue
            logits = [] if not keys else [torch.cat(keys) @ q[b, h].float() * scale]
            if sink is not None:
                logits.append(
                    (q[b, h].float() * sink[h].float()).sum().reshape(1) * scale
                )
            weights = torch.softmax(torch.cat(logits), dim=0)
            if values:
                out[b, h] = (
                    (
                        weights[: sum(x.shape[0] for x in values), None]
                        * torch.cat(values)
                    )
                    .sum(dim=0)
                    .to(q.dtype)
                )
    return out
