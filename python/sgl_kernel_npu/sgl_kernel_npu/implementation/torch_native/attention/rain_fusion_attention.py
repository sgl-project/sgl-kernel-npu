"""Pure PyTorch reference for the RainFusionAttention operator."""

import math

import torch


@torch.no_grad()
def rain_fusion_attention(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    blockshape,
    attn_mask=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    block_table=None,
    q_input_layout="TND",
    kv_input_layout="TND",
    head_num=1,
    mask_type=0,
    scale=1.0,
    inner_precise=1,
    block_size=0,
):
    """Compute block-sparse attention selected by select_idx/select_num_idx.

    Only mask_type=0 (no mask) is implemented, matching the device Kernel. The
    reference runs in FP32 and casts attention_out back to the query dtype.
    softmax_lse is never written by the device Kernel, so a zero tensor with the
    allocated shape is returned for signature compatibility.
    """
    if q_input_layout != kv_input_layout:
        raise ValueError(
            "q_input_layout and kv_input_layout must match (both TND or both BNSD)"
        )
    if attn_mask is not None:
        raise NotImplementedError("torch_native does not apply attn_mask")
    if block_table is not None:
        raise NotImplementedError("torch_native does not support block_table")
    if mask_type != 0:
        raise NotImplementedError("torch_native implements mask_type=0 (no mask) only")
    if len(blockshape) < 2:
        raise ValueError("blockshape must contain at least two elements [x, y]")
    block_x, block_y = int(blockshape[0]), int(blockshape[1])
    if block_x <= 0 or block_y <= 0:
        raise ValueError("blockshape entries must be greater than 0")

    if q_input_layout not in ("TND", "BNSD"):
        raise ValueError("q_input_layout must be TND or BNSD")
    if query.dim() != (3 if q_input_layout == "TND" else 4):
        raise ValueError(f"{q_input_layout} query has an invalid rank")
    if key.dim() != query.dim() or value.dim() != query.dim():
        raise ValueError("query/key/value must share the same layout rank")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query/key/value must be on the same device")

    q_heads, kv_heads = query.shape[1], key.shape[1]
    if head_num != kv_heads:
        raise ValueError("head_num must match the key/value head count")
    if kv_heads == 0 or q_heads % kv_heads != 0:
        raise ValueError("query heads must be a multiple of key/value heads")

    if q_input_layout == "TND":
        if actual_seq_qlen is None or actual_seq_kvlen is None:
            raise ValueError("TND requires actual_seq_qlen and actual_seq_kvlen")
        q_lens = [int(length) for length in actual_seq_qlen]
        kv_lens = [int(length) for length in actual_seq_kvlen]
        if not q_lens or len(q_lens) != len(kv_lens):
            raise ValueError("actual sequence length lists must be nonempty and equal")
        if sum(q_lens) != query.shape[0] or sum(kv_lens) != key.shape[0]:
            raise ValueError(
                "sum of sequence lengths must match the packed token count"
            )
        batch = len(q_lens)
    else:
        batch = query.shape[0]
        if key.shape[0] != batch:
            raise ValueError("query and key/value must share the batch dimension")
        q_lens = _batch_lengths(
            actual_seq_qlen, batch, query.shape[2], "actual_seq_qlen"
        )
        kv_lens = _batch_lengths(
            actual_seq_kvlen, batch, key.shape[2], "actual_seq_kvlen"
        )

    if select_idx.dim() != 3 or select_idx.shape[1] != q_heads:
        raise ValueError("select_idx must be [q_blocks, query_heads, kv_blocks]")
    if select_num_idx.dim() != 2 or select_num_idx.shape[1] != q_heads:
        raise ValueError("select_num_idx must be [q_blocks, query_heads]")
    if select_idx.shape[0] != select_num_idx.shape[0]:
        raise ValueError("select_idx and select_num_idx block counts must match")

    needed_rows = sum(math.ceil(length / block_x) for length in q_lens) * q_heads
    if needed_rows > select_idx.shape[0] * q_heads:
        raise ValueError("select_idx has fewer rows than the query blocks require")

    out = torch.zeros_like(query)
    group = q_heads // kv_heads
    max_blocks = select_idx.shape[-1]
    blocks_flat = select_idx.detach().reshape(-1, max_blocks).to("cpu")
    counts_flat = select_num_idx.detach().reshape(-1).to("cpu")

    q_offset = kv_offset = 0
    flat_row = 0
    for batch_index in range(batch):
        q_len, kv_len = q_lens[batch_index], kv_lens[batch_index]
        q_batch, k_batch, v_batch, out_batch = _batch_views(
            q_input_layout,
            query,
            key,
            value,
            out,
            batch_index,
            q_offset,
            kv_offset,
            q_len,
            kv_len,
        )
        for block_index in range(math.ceil(q_len / block_x)):
            start = block_index * block_x
            end = min(start + block_x, q_len)
            q_block = q_batch[start:end].float()
            for head in range(q_heads):
                row = flat_row
                flat_row += 1
                count = int(counts_flat[row].item())
                if count <= 0:
                    continue
                if count > max_blocks:
                    raise ValueError("select_num_idx exceeds the select_idx width")
                kv_head = head // group
                selected = blocks_flat[row, :count].tolist()
                token_ids = []
                for kv_block in selected:
                    block_start = int(kv_block) * block_y
                    if block_start < 0 or block_start >= kv_len:
                        raise ValueError("select_idx contains an out-of-range block id")
                    token_ids.append(
                        torch.arange(
                            block_start,
                            min(block_start + block_y, kv_len),
                            device=query.device,
                        )
                    )
                if not token_ids:
                    continue
                token_ids = torch.cat(token_ids)
                k_block = k_batch.index_select(0, token_ids)[:, kv_head, :].float()
                v_block = v_batch.index_select(0, token_ids)[:, kv_head, :].float()
                logits = q_block[:, head, :] @ k_block.transpose(0, 1)
                probs = torch.softmax(logits * scale, dim=-1)
                out_batch[start:end, head] = (probs @ v_block).to(query.dtype)
        q_offset += q_len
        kv_offset += kv_len

    lse_shape = query.shape[:-1] if query.dim() == 4 else query.shape
    softmax_lse = torch.zeros(lse_shape, dtype=query.dtype, device=query.device)
    return out, softmax_lse


def _batch_lengths(lengths, batch, maximum, name):
    """Return per-batch lengths; a missing or mismatched list uses the full size."""
    if lengths is None or len(lengths) != batch:
        return [maximum] * batch
    resolved = [int(length) for length in lengths]
    if any(length < 0 or length > maximum for length in resolved):
        raise ValueError(f"{name} entries must be within [0, {maximum}]")
    return resolved


def _batch_views(
    layout, query, key, value, out, index, q_offset, kv_offset, q_len, kv_len
):
    """Return [tokens, heads, dim] views for one batch in either layout."""
    if layout == "TND":
        return (
            query[q_offset : q_offset + q_len],
            key[kv_offset : kv_offset + kv_len],
            value[kv_offset : kv_offset + kv_len],
            out[q_offset : q_offset + q_len],
        )
    return (
        query[index, :, :q_len, :].transpose(0, 1),
        key[index, :, :kv_len, :].transpose(0, 1),
        value[index, :, :kv_len, :].transpose(0, 1),
        out[index, :, :q_len, :].transpose(0, 1),
    )
