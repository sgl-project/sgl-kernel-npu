"""Independent FP32 reference for ratio-4 LightningIndexer prefill selection.

This implementation scores every query against every compressed key directly
and sorts the result. It shares no code with the CANN adaptation: there is no
query grouping, no page table and no sequence-length metadata beyond the causal
horizon, so it is usable as an accuracy reference.

Rows are processed in chunks because the per-head dot product is proportional
to rows x 4 x keys; a full prefill chunk would need terabytes at once.

Ties are broken towards the lower key index, which makes the reference
deterministic. The CANN operator resolves ties by accumulation order instead,
so callers comparing the two must compare the selected set and the
non-increasing score order rather than the exact permutation of tied keys.
"""

import torch

QUERY_HEADS = 4
HEAD_DIM = 128
SELECTED_KEYS = 512
COMPRESSION_RATIO = 4
# Bound the [rows, 4, keys] float32 dot product per chunk to roughly 64 MiB.
_CHUNK_BYTES = 64 << 20


def _chunk_rows(keys):
    return max(1, _CHUNK_BYTES // (QUERY_HEADS * keys * 4))


def visible_key_counts(rows, sequence_length, device):
    """Int64 count of compressed keys each query row may causally attend to."""
    return (
        torch.arange(rows, device=device) + (sequence_length - rows + 1)
    ) // COMPRESSION_RATIO


def prefill_scores(q, k, sequence_length, begin, end):
    """FP32 ``[end - begin, keys]`` scores for a row range, invisible keys -inf.

    ``score[i, j] = sum_h relu(q[i, h] . k[j])`` with unit head weights.
    """
    keys = k.shape[0]
    dot = torch.einsum("qhd,kd->qhk", q[begin:end].float(), k[:, 0].float())
    score = dot.relu().sum(dim=1)
    counts = visible_key_counts(q.shape[0], sequence_length, q.device)[begin:end]
    visible = torch.arange(keys, device=q.device).unsqueeze(0) < counts.unsqueeze(1)
    return score.masked_fill(~visible, -torch.inf), counts


def lightning_indexer_prefill(q, k, sequence_length):
    """Ordered int32 ``[rows, 512]`` key indices, padded with -1."""
    rows, keys = q.shape[0], k.shape[0]
    out = torch.full((rows, SELECTED_KEYS), -1, dtype=torch.int32, device=q.device)
    if not rows or not keys:
        return out
    width = min(keys, SELECTED_KEYS)
    ranks = torch.arange(width, device=q.device).unsqueeze(0)
    for begin in range(0, rows, _chunk_rows(keys)):
        end = min(begin + _chunk_rows(keys), rows)
        score, counts = prefill_scores(q, k, sequence_length, begin, end)
        # A stable descending sort keeps equal scores in ascending key order.
        order = torch.sort(score, dim=-1, descending=True, stable=True).indices
        order = order[:, :width].to(torch.int32)
        out[begin:end, :width] = torch.where(ranks < counts.unsqueeze(1), order, -1)
    return out
