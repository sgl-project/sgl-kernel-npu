"""Independent CPU reference for sparse attention with shared KV.

This is an accuracy reference, not a graph-compatible execution fallback. It
covers the sliding-window (SWA) template on paged KV (``layout_kv='PA_ND'``),
which is the path exercised by the legacy operator test. FP32 is used for all
intermediates and the result is cast back to the query dtype.
"""

from typing import Optional, Tuple

import torch


def _page_cache(
    cache: torch.Tensor, block_table: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    """Gather logical [S2, N2, D] rows from [blocks, block_size, N2, D] pages."""
    block_size = cache.shape[1]
    blocks = (sequence_length + block_size - 1) // block_size
    pages = [
        cache[int(block_table[i])].reshape(block_size, cache.shape[2], cache.shape[3])
        for i in range(blocks)
    ]
    return torch.cat(pages, dim=0)[:sequence_length]


def sparse_attn_sharedkv(
    q: torch.Tensor,
    *,
    ori_kv: Optional[torch.Tensor] = None,
    cmp_kv: Optional[torch.Tensor] = None,
    ori_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 0.0,
    cmp_ratio: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 128,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate SWA/PA_ND attention and return (attn_out, empty softmax_lse)."""
    if layout_kv != "PA_ND":
        raise NotImplementedError("The reference implements layout_kv='PA_ND' only")
    if cmp_kv is not None and cmp_kv.numel() > 0:
        raise NotImplementedError(
            "The reference does not implement the compressed-KV templates"
        )
    if ori_mask_mode != 4:
        raise NotImplementedError(
            "The reference implements ori_mask_mode=4 (sliding window) only"
        )
    if return_softmax_lse:
        raise NotImplementedError("The reference does not return softmax_lse")
    if ori_block_table is None or seqused_kv is None:
        raise ValueError("PA_ND requires ori_block_table and seqused_kv")
    if sinks is None:
        raise ValueError("sinks is required")

    dtype = q.dtype
    device = q.device
    scale = float(softmax_scale)
    query = q.detach().cpu().float()
    cache = ori_kv.detach().cpu().float()
    block_table = ori_block_table.detach().cpu().long()
    seq_kv = seqused_kv.detach().cpu().long()
    sink_values = sinks.detach().cpu().float()

    tnd = layout_q == "TND"
    if tnd:
        heads, head_dim = query.shape[1], query.shape[2]
        batch = block_table.shape[0]
        cu_q = cu_seqlens_q.detach().cpu().long()
        q_lengths = [(cu_q[i + 1] - cu_q[i]).item() for i in range(batch)]
        out = torch.empty_like(query)
    else:
        batch, q_len, heads, head_dim = query.shape
        q_lengths = [q_len] * batch
        out = torch.empty_like(query)

    n2 = cache.shape[2]
    group = heads // n2
    if heads % n2 != 0:
        raise ValueError("query heads must be divisible by KV heads")

    for b in range(batch):
        kv = _page_cache(cache, block_table[b], int(seq_kv[b]))
        sequence_length = kv.shape[0]
        q_len = q_lengths[b]
        q_start = int(cu_q[b]) if tnd else 0
        for s in range(q_len):
            # Align the last query token with the last KV token.
            aligned = sequence_length - q_len + s
            left = max(aligned - ori_win_left, 0)
            right = min(aligned + ori_win_right, sequence_length - 1)
            visible = kv[left : right + 1]
            for head in range(heads):
                q_token = query[q_start + s, head] if tnd else query[b, s, head]
                k = visible[:, head // group, :]
                scores = torch.matmul(q_token, k.transpose(0, 1)) * scale
                probs = torch.softmax(
                    torch.cat([sink_values[head].reshape(1), scores]), dim=-1
                )
                value = torch.matmul(probs[1:], k)
                if tnd:
                    out[q_start + s, head] = value
                else:
                    out[b, s, head] = value

    empty_lse = torch.empty(0, dtype=torch.float32, device="cpu")
    return out.to(dtype).to(device), empty_lse.to(device)
