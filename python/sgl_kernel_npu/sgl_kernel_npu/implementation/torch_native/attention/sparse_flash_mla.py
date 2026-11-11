"""Mathematical sparse Flash MLA reference with FP32 attention arithmetic."""

import math

import torch


def _batch_lengths(batch, total, cu_seqlens, seqused, name):
    """Resolve one valid length per batch from boundaries or used lengths."""
    if seqused is not None:
        lengths = seqused.tolist()
        if len(lengths) != batch:
            raise ValueError(f"{name}: seqused length must match the batch size")
        return lengths
    if cu_seqlens is not None:
        bounds = cu_seqlens.tolist()
        if len(bounds) != batch + 1 or bounds[0] != 0:
            raise ValueError(f"{name}: cu_seqlens must start at 0 and cover the batch")
        return [end - start for start, end in zip(bounds, bounds[1:])]
    return [total] * batch


def _query_batches(q, layout_q, cu_seqlens_q, seqused_q):
    """Return per-batch [N, S, D] FP32 queries."""
    if layout_q == "TND":
        if cu_seqlens_q is None:
            raise ValueError("TND queries require cu_seqlens_q")
        bounds = cu_seqlens_q.tolist()
        if bounds[0] != 0 or bounds[-1] != q.shape[0]:
            raise ValueError("cu_seqlens_q must span the packed queries")
        parts = [q[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
        if seqused_q is not None:
            used = seqused_q.tolist()
            if used != [part.shape[0] for part in parts]:
                raise NotImplementedError(
                    "torch_native requires fully used query segments"
                )
        return [part.transpose(0, 1).float() for part in parts]
    if layout_q != "BSND" or q.ndim != 4:
        raise NotImplementedError("torch_native supports BSND or TND queries")
    lengths = _batch_lengths(q.shape[0], q.shape[1], None, seqused_q, "q")
    return [
        q[batch, :length].transpose(0, 1).float()
        for batch, length in enumerate(lengths)
    ]


def _kv_batches(kv, layout_kv, cu_seqlens, seqused, block_table, name):
    """Return per-batch [N, S, D] FP32 shared K/V."""
    if layout_kv == "PA_BBND":
        if block_table is None or seqused is None:
            raise ValueError(f"{name}: PA_BBND requires a block table and seqused")
        lengths = seqused.tolist()
        if len(lengths) != block_table.shape[0]:
            raise ValueError(f"{name}: seqused length must match the block table")
        page_size, heads, width = kv.shape[1], kv.shape[2], kv.shape[3]
        parts = []
        for batch, length in enumerate(lengths):
            pages = math.ceil(length / page_size)
            if length <= 0 or pages > block_table.shape[1]:
                raise ValueError(f"{name}: invalid length or insufficient pages")
            ids = block_table[batch, :pages].to(torch.int64)
            flat = kv.index_select(0, ids).reshape(-1, heads, width)
            parts.append(flat[:length].transpose(0, 1).float())
        return parts
    if layout_kv == "BSND" and kv.ndim == 4:
        lengths = _batch_lengths(kv.shape[0], kv.shape[1], None, seqused, name)
        return [
            kv[batch, :length].transpose(0, 1).float()
            for batch, length in enumerate(lengths)
        ]
    if layout_kv == "TND" and kv.ndim == 3:
        if cu_seqlens is None:
            raise ValueError(f"{name}: TND KV requires cu_seqlens")
        bounds = cu_seqlens.tolist()
        lengths = [end - start for start, end in zip(bounds, bounds[1:])]
        if seqused is not None and seqused.tolist() != lengths:
            raise NotImplementedError(f"{name}: torch_native requires full TND KV")
        return [
            kv[bounds[i] : bounds[i + 1]].transpose(0, 1).float()
            for i in range(len(bounds) - 1)
        ]
    raise NotImplementedError(f"{name}: unsupported layout_kv {layout_kv!r}")


def _ori_window(index, act_q, act_kv, mask_mode, win_left, win_right):
    """Return the half-open ori_kv key range for one query position."""
    if mask_mode == 0:
        return 0, act_kv
    threshold = act_kv - act_q + index + 1
    if mask_mode == 3:
        return 0, min(max(threshold, 0), act_kv)
    if mask_mode == 4:
        start = 0 if win_left == -1 else max(threshold - win_left - 1, 0)
        end = act_kv if win_right == -1 else min(max(threshold + win_right, 0), act_kv)
        return start, end
    raise NotImplementedError("torch_native supports ori_mask_mode 0/3/4")


def _cmp_limit(index, act_q, restored, ratio, mask_mode, act_cmp):
    """Return the exclusive compressed-token limit for one query position."""
    if mask_mode == 0:
        return act_cmp
    threshold = (restored - act_q + index + 1) // ratio
    return min(max(threshold, 0), act_cmp)


def _cmp_sparse_keys(indices, limit, topk_length):
    """Pick CSA compressed token ids, stopping at the first -1 terminator."""
    count = len(indices) if topk_length is None else min(topk_length, len(indices))
    selected = []
    for value in indices[: min(count, limit)]:
        token = int(value)
        if token == -1:
            break
        if 0 <= token < limit:
            selected.append(token)
    return selected


def _softmax_with_sinks(scores, sink):
    """Return probabilities, per-row log-sum-exp and the softmax denominator."""
    row_max = scores.max(dim=-1).values
    if sink is not None:
        row_max = torch.maximum(row_max, sink)
        weights = torch.exp(scores - row_max.unsqueeze(-1))
        total = weights.sum(dim=-1) + torch.exp(sink - row_max)
    else:
        weights = torch.exp(scores - row_max.unsqueeze(-1))
        total = weights.sum(dim=-1)
    return weights / total.unsqueeze(-1), row_max + torch.log(total), total


def sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
):
    """Compute softmax(scale * QK^T)V over shared original and compressed KV.

    Supports the SWA, CSA and HCA scenarios with FP32 arithmetic on the input
    device. Metadata is a backend scheduling artifact and does not affect the
    math. Host reads of lengths and page IDs make this a reference, not a
    graph-safe kernel.
    """
    if ori_sparse_indices is not None or ori_topk_length is not None:
        raise NotImplementedError("torch_native does not implement ori sparse indices")
    if topk_value_mode != 1:
        raise NotImplementedError("torch_native only implements topk_value_mode 1")
    if ori_kv is None:
        raise ValueError("ori_kv is required")
    if cmp_kv is None and (cmp_sparse_indices is not None or cmp_mask_mode != 0):
        raise ValueError("cmp_kv is required for CSA and HCA")
    if cmp_kv is None and cmp_ratio != 1:
        raise ValueError("cmp_ratio must be 1 without cmp_kv")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("q must be a floating-point tensor")
    for name, tensor in (("ori_kv", ori_kv), ("cmp_kv", cmp_kv), ("sinks", sinks)):
        if tensor is not None and tensor.device != q.device:
            raise ValueError(f"{name} must be on the query device")
    if ori_kv.dtype != q.dtype or (cmp_kv is not None and cmp_kv.dtype != q.dtype):
        raise ValueError("q and shared K/V must share one dtype")

    q_batches = _query_batches(q, layout_q, cu_seqlens_q, seqused_q)
    ori_list = _kv_batches(
        ori_kv, layout_kv, cu_seqlens_ori_kv, seqused_ori_kv, ori_block_table, "ori_kv"
    )
    cmp_list = (
        _kv_batches(
            cmp_kv,
            layout_kv,
            cu_seqlens_cmp_kv,
            seqused_cmp_kv,
            cmp_block_table,
            "cmp_kv",
        )
        if cmp_kv is not None
        else None
    )
    batch_size = len(q_batches)
    num_heads_q = q_batches[0].shape[0]
    num_heads_kv = ori_list[0].shape[0]
    if num_heads_q % num_heads_kv:
        raise ValueError("num_heads_q must be a multiple of num_heads_kv")
    group = num_heads_q // num_heads_kv
    head_dim = q_batches[0].shape[-1]
    residual = cmp_residual_kv.tolist() if cmp_residual_kv is not None else None
    if residual is not None and len(residual) != batch_size:
        raise ValueError("cmp_residual_kv must hold one value per batch")

    packed = layout_q == "TND"
    if packed:
        offsets_q = cu_seqlens_q.tolist()
        total_q = q.shape[0]
        output = torch.empty(
            (total_q, num_heads_q, head_dim), dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(
            (num_heads_kv, total_q, group), dtype=torch.float32, device=q.device
        )
    else:
        output = torch.zeros(q.shape, dtype=q.dtype, device=q.device)
        lse = torch.zeros(
            (batch_size, num_heads_kv, q.shape[1], group),
            dtype=torch.float32,
            device=q.device,
        )

    for batch in range(batch_size):
        queries = q_batches[batch]
        original = ori_list[batch]
        compressed = None if cmp_list is None else cmp_list[batch]
        act_q = queries.shape[1]
        act_ori = original.shape[1]
        act_cmp = 0 if compressed is None else compressed.shape[1]
        ratio = int(cmp_ratio)
        restore = 0 if residual is None else int(residual[batch])
        if cmp_mask_mode == 0 or ratio == 1:
            restore = 0
        restored = act_cmp * ratio + restore
        batch_out = torch.zeros(
            (num_heads_q, act_q, head_dim), dtype=torch.float32, device=q.device
        )
        for head_kv in range(num_heads_kv):
            keys_ori = original[head_kv]
            keys_cmp = None if compressed is None else compressed[head_kv]
            query = queries[head_kv * group : (head_kv + 1) * group]
            sink = (
                None
                if sinks is None
                else sinks[head_kv * group : (head_kv + 1) * group]
            )
            for index in range(act_q):
                position = offsets_q[batch] + index if packed else None
                start, end = _ori_window(
                    index, act_q, act_ori, ori_mask_mode, ori_win_left, ori_win_right
                )
                keys = [keys_ori[start:end]] if end > start else []
                if keys_cmp is not None:
                    limit = _cmp_limit(
                        index, act_q, restored, ratio, cmp_mask_mode, act_cmp
                    )
                    if cmp_sparse_indices is not None:
                        topk = None
                        if cmp_topk_length is not None:
                            row = position if packed else (batch, index, 0)
                            topk = int(cmp_topk_length[row].item())
                        if packed:
                            ids_row = cmp_sparse_indices[position, 0].tolist()
                        else:
                            ids_row = cmp_sparse_indices[batch, index, 0].tolist()
                        token_ids = _cmp_sparse_keys(ids_row, limit, topk)
                        picked = keys_cmp.index_select(
                            0,
                            torch.tensor(token_ids, dtype=torch.int64, device=q.device),
                        )
                    else:
                        picked = keys_cmp[:limit]
                    if picked.shape[0]:
                        keys.append(picked)
                if not keys:
                    continue
                shared = torch.cat(keys, dim=0)
                scores = torch.matmul(query[:, index], shared.transpose(0, 1))
                scores.mul_(softmax_scale)
                probabilities, row_lse, _ = _softmax_with_sinks(scores, sink)
                batch_out[head_kv * group : (head_kv + 1) * group, index] = (
                    torch.matmul(probabilities, shared)
                )
                if return_softmax_lse:
                    if packed:
                        lse[head_kv, position] = row_lse
                    else:
                        lse[batch, head_kv, index] = row_lse
        if packed:
            output[offsets_q[batch] : offsets_q[batch + 1]] = batch_out.transpose(
                0, 1
            ).to(q.dtype)
        else:
            output[batch, :act_q] = batch_out.transpose(0, 1).to(q.dtype)

    if not return_softmax_lse:
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return output, lse
