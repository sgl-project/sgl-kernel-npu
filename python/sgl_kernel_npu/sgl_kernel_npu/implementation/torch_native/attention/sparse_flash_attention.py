"""Independent CPU reference for sparse flash attention.

This is an accuracy reference, not a graph-compatible execution fallback. It
covers ``sparse_mode=0`` (explicit top-k selection), ``attention_mode=2``
(MLA/DeepSeek sparse attention) and ``sparse_block_size`` blocks, which is the
path used by the legacy operator tests. FP32 is used for all intermediates; the
attention output is cast back to the query dtype.
"""

import math
from typing import Optional, Tuple

import torch

# Matches the kernel's "no visible token" softmax maximum.
_SOFTMAX_MIN = -2e38


def _paged_sequence(
    cache: torch.Tensor, block_table: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    """Gather logical [S, N2, D] rows from [blocks, block_size, N2, D] pages."""
    block_size = cache.shape[1]
    blocks = (sequence_length + block_size - 1) // block_size
    pages = [
        cache[int(block_table[i])].reshape(block_size, cache.shape[2], cache.shape[3])
        for i in range(blocks)
    ]
    return torch.cat(pages, dim=0)[:sequence_length]


def _visible_positions(indices, threshold: int, sparse_block_size: int):
    """Expand top-k block indices into visible KV positions, matching the kernel."""
    valid_count = min(len(indices), math.ceil(threshold / sparse_block_size))
    positions = []
    for index in indices[:valid_count].tolist():
        if index == -1:
            break
        begin = index * sparse_block_size
        if begin >= threshold:
            continue
        end = min(begin + sparse_block_size, threshold)
        positions.extend(range(begin, end))
    return positions


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    *,
    block_table: Optional[torch.Tensor] = None,
    actual_seq_lengths_query: Optional[torch.Tensor] = None,
    actual_seq_lengths_kv: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    attention_mode: int = 2,
    return_softmax_lse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (attention_out, softmax_max, softmax_sum) for sparse attention."""
    if layout_query not in ("BSND", "TND"):
        raise NotImplementedError("layout_query must be 'BSND' or 'TND'")
    if layout_kv not in ("BSND", "TND", "PA_BSND"):
        raise NotImplementedError("layout_kv must be 'BSND', 'TND' or 'PA_BSND'")
    if attention_mode != 2:
        raise NotImplementedError("The reference implements attention_mode=2 only")
    if sparse_mode != 0:
        raise NotImplementedError("The reference implements sparse_mode=0 only")
    if layout_kv == "PA_BSND" and block_table is None:
        raise ValueError("layout_kv='PA_BSND' requires block_table")

    dtype = query.dtype
    device = query.device
    scale = float(scale_value)
    query_cpu = query.detach().cpu().float()
    key_cpu = key.detach().cpu().float()
    value_cpu = value.detach().cpu().float()
    topk_cpu = sparse_indices.detach().cpu().long()
    # The attention output keeps the non-rope head dimension of the value.
    out = torch.zeros(
        query_cpu.shape[:-1] + (value_cpu.shape[-1],), dtype=torch.float32
    )
    if query_rope is not None:
        query_cpu = torch.cat((query_cpu, query_rope.detach().cpu().float()), dim=-1)
    if key_rope is not None:
        key_cpu = torch.cat((key_cpu, key_rope.detach().cpu().float()), dim=-1)

    tnd = layout_query == "TND"
    if tnd:
        head_count = query_cpu.shape[1]
        tokens = query_cpu.shape[0]
        # The native TND input is a prefix sum over the query token counts.
        if actual_seq_lengths_query is not None:
            prefix = actual_seq_lengths_query.detach().cpu().long().tolist()
            query_offsets = [0] + prefix[:-1]
            query_lengths = [prefix[0]] + [
                prefix[i] - prefix[i - 1] for i in range(1, len(prefix))
            ]
        else:
            query_offsets = [0]
            query_lengths = [tokens]
        batch = len(query_lengths)
    else:
        batch, q_len, head_count, _ = query_cpu.shape
        query_lengths = [q_len] * batch
        query_offsets = [0] * batch
        tokens = 0

    kv_heads = key_cpu.shape[-2]
    group = head_count // kv_heads

    if actual_seq_lengths_kv is not None:
        kv_prefix = actual_seq_lengths_kv.detach().cpu().long().tolist()
        if layout_kv == "TND":
            kv_offsets = [0] + kv_prefix[:-1]
            kv_lens = [kv_prefix[0]] + [
                kv_prefix[i] - kv_prefix[i - 1] for i in range(1, len(kv_prefix))
            ]
        else:
            kv_offsets = [0] * len(kv_prefix)
            kv_lens = kv_prefix
        if len(kv_lens) < batch:
            kv_lens = kv_lens * batch
            kv_offsets = kv_offsets * batch
    elif layout_kv == "PA_BSND":
        kv_lens = [key_cpu.shape[0] * key_cpu.shape[1]] * batch
        kv_offsets = [0] * batch
    elif layout_kv == "BSND":
        kv_lens = [key_cpu.shape[1]] * batch
        kv_offsets = [0] * batch
    else:
        kv_lens = [key_cpu.shape[0]] * batch
        kv_offsets = [0] * batch

    if return_softmax_lse:
        lse_shape = (
            (kv_heads, tokens, group) if tnd else (batch, kv_heads, q_len, group)
        )
        softmax_max = torch.full(lse_shape, _SOFTMAX_MIN, dtype=torch.float32)
        softmax_sum = torch.zeros(lse_shape, dtype=torch.float32)
    else:
        softmax_max = torch.empty(0, dtype=torch.float32)
        softmax_sum = torch.empty(0, dtype=torch.float32)

    block_table_cpu = None if block_table is None else block_table.detach().cpu().long()
    for b in range(batch):
        if layout_kv == "PA_BSND":
            key_b = _paged_sequence(key_cpu, block_table_cpu[b], kv_lens[b])
            value_b = _paged_sequence(value_cpu, block_table_cpu[b], kv_lens[b])
        elif layout_kv == "TND":
            kv_offset = kv_offsets[b]
            key_b = key_cpu[kv_offset : kv_offset + kv_lens[b]]
            value_b = value_cpu[kv_offset : kv_offset + kv_lens[b]]
        else:
            key_b = key_cpu[b, : kv_lens[b]]
            value_b = value_cpu[b, : kv_lens[b]]
        threshold = kv_lens[b]

        for s in range(query_lengths[b]):
            token = query_offsets[b] + s
            query_row = query_cpu[token] if tnd else query_cpu[b, s]
            for n2 in range(kv_heads):
                indices = topk_cpu[token, n2] if tnd else topk_cpu[b, s, n2]
                positions = _visible_positions(indices, threshold, sparse_block_size)
                head_slice = slice(n2 * group, (n2 + 1) * group)
                if not positions:
                    continue
                keys = key_b[positions, n2]
                values = value_b[positions, n2]
                scores = query_row[head_slice] @ keys.transpose(0, 1) * scale
                row_max = scores.max(dim=-1).values
                probs = torch.exp(scores - row_max.unsqueeze(-1))
                row_sum = probs.sum(dim=-1)
                value_out = (probs @ values) / row_sum.unsqueeze(-1)
                if tnd:
                    out[token, head_slice] = value_out
                    if return_softmax_lse:
                        softmax_max[n2, token] = row_max
                        softmax_sum[n2, token] = row_sum
                else:
                    out[b, s, head_slice] = value_out
                    if return_softmax_lse:
                        softmax_max[b, n2, s] = row_max
                        softmax_sum[b, n2, s] = row_sum

    return out.to(dtype).to(device), softmax_max.to(device), softmax_sum.to(device)
