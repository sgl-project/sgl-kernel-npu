# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 PyTorch reference for fixed-width paged GQA."""

import torch


def paged_mtp_attention_decode(
    *,
    query,
    k_cache,
    v_cache,
    seq_lens,
    cu_q_lens,
    block_tables,
    page_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    gamma,
    softmax_scale,
    max_kv_len,
    causal=False,
):
    """Gather valid pages and evaluate attention independently of Triton.

    Lengths and offsets are read on the host. This correctness reference is
    intentionally unsuitable for graph capture or latency-sensitive execution.
    max_kv_len is a scheduling hint and does not limit valid KV reads.
    """
    output = torch.zeros_like(query)
    group_size = num_q_heads // num_kv_heads
    for batch, length in enumerate(seq_lens.tolist()):
        if length == 0:
            # Empty slots may have invalid page indices; never dereference them.
            continue
        q_start = int(cu_q_lens[batch].item())
        page_count = (length + page_size - 1) // page_size
        pages = block_tables[batch, :page_count].long()
        keys = k_cache[pages].reshape(-1, num_kv_heads, head_dim)[:length].float()
        values = v_cache[pages].reshape(-1, num_kv_heads, head_dim)[:length].float()
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)
        q = query[q_start : q_start + gamma].float()
        scores = torch.einsum("qhd,khd->hqk", q, keys) * softmax_scale
        if causal:
            positions = torch.arange(length, device=query.device)
            ends = length - gamma + torch.arange(gamma, device=query.device)
            allowed = positions[None, :] <= ends[:, None]
            scores = scores.masked_fill(~allowed[None, :, :], -float("inf"))
        # Short causal contexts can leave a query with no visible key.
        probabilities = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        output[q_start : q_start + gamma] = torch.einsum(
            "hqk,khd->qhd", probabilities, values
        ).to(query.dtype)
    return output
