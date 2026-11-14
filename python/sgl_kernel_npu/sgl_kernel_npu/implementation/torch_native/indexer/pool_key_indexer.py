"""Independent FP32 reference for CANN pool_key_indexer test coverage.

Supports BSND query/key tensors and TND queries with PA_BBND pooled keys. It
implements weighted ReLU scores, right-aligned causal masking, pool expansion,
and incomplete-tail append. This is a correctness reference, not a serving
backend.
"""

import torch


def _host_ints(value):
    if value is None:
        return None
    return value.detach().cpu().tolist()


def _sequence_ends(ends, total, name):
    if ends is None:
        raise ValueError(f"{name} is required for TND queries")
    if (
        not ends
        or ends[-1] != total
        or any(end < start for start, end in zip(ends, ends[1:]))
    ):
        raise ValueError(f"{name} must be nondecreasing and end at {total}")
    return [0, *ends]


def _scores(query, pool_key, weights, *, mask_mode):
    head_dim = query.shape[-1]
    score = torch.einsum("qhd,kd->qhk", query.float(), pool_key[:, 0].float())
    score.mul_(head_dim**-0.5).relu_()
    score.mul_(weights.float().unsqueeze(-1))
    score = score.sum(dim=1)
    if mask_mode == 0:
        return score
    if mask_mode != 3:
        raise NotImplementedError("torch_native supports mask_mode 0 or 3 only")

    query_length, pool_count = score.shape
    query_pos = torch.arange(query_length, device=score.device).unsqueeze(-1)
    pool_pos = torch.arange(pool_count, device=score.device)
    visible = pool_pos <= query_pos + pool_count - query_length
    return score.masked_fill(~visible, -torch.inf)


def _select(score, *, topk, pool_size, pool_tail_k, return_value):
    query_length, pool_count = score.shape
    selected_pools = topk // pool_size
    result = torch.full(
        (query_length, topk + pool_size - 1),
        -1,
        dtype=torch.int32,
        device=score.device,
    )
    sparse_values = (
        score.new_full((query_length, selected_pools), -torch.inf)
        if return_value
        else score.new_empty((0,))
    )
    width = min(selected_pools, pool_count)
    if width:
        values, pools = score.topk(width, dim=-1)
        token_offsets = torch.arange(pool_size, device=score.device)
        expanded = pools.unsqueeze(-1) * pool_size + token_offsets
        expanded = expanded.masked_fill(values.unsqueeze(-1).isneginf(), -1)
        result[:, : width * pool_size] = expanded.reshape(query_length, -1).to(
            torch.int32
        )
        if return_value:
            sparse_values[:, :width] = values
    if not 0 <= pool_tail_k < pool_size:
        raise ValueError("pool_tail_k must be in [0, pool_size)")
    if pool_tail_k:
        tail = pool_count * pool_size + torch.arange(pool_tail_k, device=score.device)
        result[:, topk : topk + pool_tail_k] = tail.to(torch.int32)
    return result, sparse_values


def _paged_pool_keys(pool_key, block_table, request, pool_count):
    if block_table is None:
        raise ValueError("PA_BBND requires block_table")
    blocks = block_table[request].to(torch.long)
    pages = pool_key.index_select(0, blocks).flatten(0, 1)
    if pool_count > pages.shape[0]:
        raise ValueError("actual_seq_k exceeds paged pool_key capacity")
    return pages[:pool_count]


def pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    topk=2048,
    pool_size=16,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
):
    """Return (indices, values) using FP32 scoring.

    quant_mode is accepted for call compatibility. The reference scores the
    supplied tensor values without reproducing vendor quantized rounding.
    """
    if topk <= 0 or pool_size <= 0 or topk % pool_size:
        raise ValueError("topk must be a positive multiple of pool_size")
    tails = _host_ints(pool_tail_k)

    if layout_q == "BSND" and layout_k == "BSND":
        if query.ndim != 4 or pool_key.ndim != 4 or weights.ndim != 3:
            raise ValueError("BSND requires query/key/weights ranks 4/4/3")
        if query.shape[0] != pool_key.shape[0] or query.shape[:-1] != weights.shape:
            raise ValueError(
                "BSND query, pool_key, and weights batch dimensions differ"
            )
        if len(tails) != query.shape[0]:
            raise ValueError("pool_tail_k must provide one value per batch item")
        output = [
            _select(
                _scores(
                    query[batch],
                    pool_key[batch],
                    weights[batch],
                    mask_mode=mask_mode,
                ),
                topk=topk,
                pool_size=pool_size,
                pool_tail_k=tails[batch],
                return_value=return_value,
            )
            for batch in range(query.shape[0])
        ]

    elif layout_q == "TND" and layout_k in {"TND", "PA_BBND"}:
        if query.ndim != 3 or weights.ndim != 2 or query.shape[:-1] != weights.shape:
            raise ValueError(
                "TND requires query/weights ranks 3/2 with matching prefixes"
            )
        query_ends = _host_ints(actual_seq_q)
        query_ends = _sequence_ends(query_ends, query.shape[0], "actual_seq_q")
        key_ends = _host_ints(actual_seq_k)
        if key_ends is None or len(key_ends) != len(query_ends) - 1:
            raise ValueError("actual_seq_k must provide one value per request")
        if len(tails) != len(key_ends):
            raise ValueError("pool_tail_k must provide one value per request")

        output = []
        previous_key_end = 0
        for request, (query_start, query_end, pool_count) in enumerate(
            zip(query_ends, query_ends[1:], key_ends)
        ):
            if pool_count < 0:
                raise ValueError("actual_seq_k must be nonnegative")
            if layout_k == "PA_BBND":
                keys = _paged_pool_keys(pool_key, block_table, request, pool_count)
            else:
                if pool_count < previous_key_end or pool_count > pool_key.shape[0]:
                    raise ValueError("TND actual_seq_k must be cumulative and in range")
                keys = pool_key[previous_key_end:pool_count]
                previous_key_end = pool_count
            output.append(
                _select(
                    _scores(
                        query[query_start:query_end],
                        keys,
                        weights[query_start:query_end],
                        mask_mode=mask_mode,
                    ),
                    topk=topk,
                    pool_size=pool_size,
                    pool_tail_k=tails[request],
                    return_value=return_value,
                )
            )
    else:
        raise NotImplementedError(
            "torch_native supports BSND/BSND and TND/{TND, PA_BBND} only"
        )
    join = torch.stack if layout_q == "BSND" else torch.cat
    return (
        join([indices for indices, _ in output]),
        (
            join([values for _, values in output])
            if return_value
            else query.new_empty((0,), dtype=torch.float32)
        ),
    )
