"""Independent unquantized FIA reference shared by both SDK schemas."""

from itertools import pairwise

import torch

from .sparse_flash_attention import _paged_sequence


def dense_attention(query, key, value, scale, *, mask=None, bias=None):
    """Compute FP32 attention for [B, H, S, D] tensors."""
    repeats = query.size(1) // key.size(1)
    if query.size(1) % key.size(1) or key.size(1) != value.size(1):
        raise ValueError("Query heads must be a multiple of the key/value heads")
    key = key.float().repeat_interleave(repeats, dim=1)
    value = value.float().repeat_interleave(repeats, dim=1)
    scores = (query.float() @ key.transpose(-1, -2)) * scale
    if bias is not None:
        scores = scores + bias.float()
    if mask is not None:
        if mask.dtype in (torch.bool, torch.uint8):
            scores = scores.masked_fill(mask.bool(), -torch.inf)
        else:
            scores = scores + mask.float()
    # Softmax of a fully masked row is NaN in PyTorch. FIA returns zero there.
    probabilities = scores.softmax(dim=-1)
    probabilities = probabilities.masked_fill(
        scores.isneginf().all(dim=-1, keepdim=True), 0
    )
    return (probabilities @ value).to(query.dtype), scores.logsumexp(
        dim=-1, keepdim=True
    )


def _tnd_attention(q, k, v, scale, q_lengths, kv_lengths):
    q_ends = [0, *q_lengths]
    kv_ends = [0, *kv_lengths]
    if (
        len(q_ends) != len(kv_ends)
        or (q_ends[-1], kv_ends[-1]) != (q.size(0), k.size(0))
        or any(left > right for left, right in pairwise(q_ends))
        or any(left > right for left, right in pairwise(kv_ends))
    ):
        raise ValueError(
            "TND cumulative lengths must be nondecreasing and cover the query and KV tokens"
        )
    output = q.new_zeros((*q.shape[:2], v.size(-1)))
    lse = output.new_full((*q.shape[:2], 1), -torch.inf, dtype=torch.float32)
    for (q_start, q_end), (kv_start, kv_end) in zip(
        pairwise(q_ends), pairwise(kv_ends)
    ):
        if q_end == q_start or kv_end == kv_start:
            continue
        result, row_lse = dense_attention(
            q[q_start:q_end].transpose(0, 1).unsqueeze(0),
            k[kv_start:kv_end].transpose(0, 1).unsqueeze(0),
            v[kv_start:kv_end].transpose(0, 1).unsqueeze(0),
            scale,
        )
        output[q_start:q_end] = result[0].transpose(0, 1)
        lse[q_start:q_end] = row_lse[0].transpose(0, 1)
    return output, lse


def _paged_bsnd_decode(q, k, v, block_table, block_size, kv_lengths, kv_heads, scale):
    if q.size(1) != 1 or k.ndim != 3 or v.ndim != 3:
        raise NotImplementedError(
            "Paged FIA requires single-token BSND queries and flattened KV pages"
        )
    if (
        block_size <= 0
        or k.shape[:2] != v.shape[:2]
        or k.size(1) != block_size
        or block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.size(0) != q.size(0)
        or len(kv_lengths) != q.size(0)
    ):
        raise ValueError("Invalid flattened paged BSND decode shapes or metadata")
    k = k.unflatten(-1, (kv_heads, -1))
    v = v.unflatten(-1, (kv_heads, -1))
    output = q.new_zeros((*q.shape[:-1], v.size(-1)))
    lse = output.new_full((q.size(0), q.size(2), 1, 1), -torch.inf, dtype=torch.float32)
    for index, (table, length) in enumerate(zip(block_table, kv_lengths)):
        if not 0 <= length <= table.numel() * block_size:
            raise ValueError("KV length exceeds the capacity of the block table")
        if length == 0:
            continue
        used_pages = table[: (length + block_size - 1) // block_size]
        if ((used_pages < 0) | (used_pages >= k.size(0))).any():
            raise ValueError("Active FIA block-table entries must index the KV pages")
        result, row_lse = dense_attention(
            q[index : index + 1].transpose(1, 2),
            _paged_sequence(k, table, length).transpose(0, 1).unsqueeze(0),
            _paged_sequence(v, table, length).transpose(0, 1).unsqueeze(0),
            scale,
        )
        output[index] = result[0].transpose(0, 1)
        lse[index] = row_lse[0]
    return output, lse


def fused_infer_attention_score(
    query,
    key,
    value,
    *,
    num_heads=1,
    num_key_value_heads=0,
    scale=1.0,
    input_layout="BSH",
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    block_table=None,
    block_size=0,
    query_rope=None,
    key_rope=None,
    atten_mask=None,
    pse_shift=None,
    sparse_mode=0,
    inner_precise=0,
    softmax_lse_flag=False,
    pre_tokens=2147483647,
    next_tokens=2147483647,
):
    """Evaluate dense attention, packed TND prefix attention, or paged decode."""
    if (
        sparse_mode not in (0, 1)
        or inner_precise not in (0, 1, 2, 3)
        or (pre_tokens, next_tokens) != (2147483647, 2147483647)
    ):
        raise NotImplementedError("torch_native FIA supports only the default windows")
    if query.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != query.dtype for tensor in (key, value)
    ):
        raise NotImplementedError("torch_native requires unquantized FP16/BF16 Q/K/V")
    q, k, v = (tensor.detach().cpu() for tensor in (query, key, value))
    if block_table is not None or input_layout == "TND":
        if sparse_mode != 0 or atten_mask is not None or pse_shift is not None:
            raise NotImplementedError("Paged decode and TND require unmasked FIA")
        kv_heads = num_key_value_heads or num_heads
        if block_table is not None:
            if (
                input_layout != "BSND"
                or actual_seq_lengths is not None
                or actual_seq_lengths_kv is None
                or query_rope is not None
                or key_rope is not None
            ):
                raise NotImplementedError("Paged FIA requires BSND GQA decode")
            if q.ndim != 4 or q.size(2) != num_heads:
                raise ValueError("Query heads must match the BSND query")
            result, lse = _paged_bsnd_decode(
                q,
                k,
                v,
                block_table.detach().cpu(),
                block_size,
                list(actual_seq_lengths_kv),
                kv_heads,
                scale,
            )
        else:
            if actual_seq_lengths is None or actual_seq_lengths_kv is None:
                raise ValueError("TND requires cumulative query and KV lengths")
            if not (q.ndim == k.ndim == v.ndim == 3) or k.size(0) != v.size(0):
                raise ValueError(
                    "TND requires rank-three tensors and matching K/V lengths"
                )
            if (q.size(1), k.size(1), v.size(1)) != (num_heads, kv_heads, kv_heads):
                raise ValueError("Head counts must match the TND tensors")
            if (query_rope is None) != (key_rope is None):
                raise ValueError("Query and key RoPE tensors must be supplied together")
            if query_rope is not None:
                if (
                    query_rope.shape[:-1] != q.shape[:-1]
                    or key_rope.shape[:-1] != k.shape[:-1]
                    or query_rope.size(-1) != key_rope.size(-1)
                    or query_rope.dtype != q.dtype
                    or key_rope.dtype != k.dtype
                ):
                    raise ValueError(
                        "TND RoPE tensors must match their Q/K shapes and dtype"
                    )
                q = torch.cat((q, query_rope.detach().cpu()), dim=-1)
                k = torch.cat((k, key_rope.detach().cpu()), dim=-1)
            result, lse = _tnd_attention(
                q, k, v, scale, actual_seq_lengths, actual_seq_lengths_kv
            )
        if not softmax_lse_flag:
            lse = lse.new_empty(0)
        return result.to(query.device), lse.to(query.device)
    if query_rope is not None or key_rope is not None:
        raise NotImplementedError("torch_native FIA RoPE requires TND")
    if input_layout == "BSH":
        head_dim = q.size(-1) // num_heads
        kv_heads = num_key_value_heads or k.size(-1) // head_dim
        q, k, v = (
            tensor.unflatten(-1, (heads, -1)).transpose(1, 2)
            for tensor, heads in zip((q, k, v), (num_heads, kv_heads, kv_heads))
        )
    elif input_layout == "BSND":
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    elif input_layout != "BNSD":
        raise NotImplementedError("torch_native supports BSH, BSND and BNSD FIA")
    if q.size(1) != num_heads or num_key_value_heads not in (0, k.size(1)):
        raise ValueError("Head counts must match the query and key layouts")
    batch, heads, q_size, _ = q.shape
    kv_size = k.size(-2)
    q_lengths = (
        [q_size] * batch if actual_seq_lengths is None else list(actual_seq_lengths)
    )
    kv_lengths = (
        [kv_size] * batch
        if actual_seq_lengths_kv is None
        else list(actual_seq_lengths_kv)
    )
    if len(q_lengths) != batch or len(kv_lengths) != batch:
        raise ValueError("Sequence lengths must contain one entry per batch item")
    mask = atten_mask.detach().cpu() if atten_mask is not None else None
    bias = pse_shift.detach().cpu() if pse_shift is not None else None
    if mask is not None:
        mask = torch.broadcast_to(
            mask[..., :q_size, :kv_size], (batch, heads, q_size, kv_size)
        )
    if bias is not None:
        bias = torch.broadcast_to(
            bias[..., :q_size, :kv_size], (batch, heads, q_size, kv_size)
        )
    result = q.new_zeros((batch, heads, q_size, v.size(-1)))
    lse = torch.full(
        (batch, heads, q_size, 1), -torch.inf, dtype=torch.float32, device="cpu"
    )
    for index, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        if not (0 <= q_len <= q_size and 0 <= kv_len <= kv_size):
            raise ValueError("Actual sequence lengths must fit the tensor lengths")
        if q_len == 0 or kv_len == 0:
            continue
        rows = slice(index, index + 1)
        result[rows, :, :q_len], lse[rows, :, :q_len] = dense_attention(
            q[rows, :, :q_len],
            k[rows, :, :kv_len],
            v[rows, :, :kv_len],
            scale,
            mask=mask[rows, :, :q_len, :kv_len] if mask is not None else None,
            bias=bias[rows, :, :q_len, :kv_len] if bias is not None else None,
        )
    if input_layout != "BNSD":
        result = result.transpose(1, 2).contiguous()
        if input_layout == "BSH":
            result = result.flatten(2)
    if not softmax_lse_flag:
        lse = lse.new_empty(0)
    return result.to(query.device), lse.to(query.device)


def fused_infer_attention_score_v2(
    query,
    key,
    value,
    *,
    num_query_heads=1,
    softmax_scale=1.0,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    return_softmax_lse=False,
    **kwargs,
):
    return fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=num_query_heads,
        scale=softmax_scale,
        actual_seq_lengths=actual_seq_qlen,
        actual_seq_lengths_kv=actual_seq_kvlen,
        softmax_lse_flag=return_softmax_lse,
        **kwargs,
    )


def _write_out(out, result):
    dst, dst_lse = out
    for buffer, tensor in zip((dst, dst_lse), result):
        if (buffer.shape, buffer.dtype, buffer.device) != (
            tensor.shape,
            tensor.dtype,
            tensor.device,
        ):
            raise ValueError("FIA out buffers must match the result")
    dst.copy_(result[0])
    dst_lse.copy_(result[1])
    return dst, dst_lse


def fused_infer_attention_score_out(
    query, key, value, *, out, workspace=None, **kwargs
):
    return _write_out(out, fused_infer_attention_score(query, key, value, **kwargs))


def fused_infer_attention_score_v2_out(
    query, key, value, *, out, workspace=None, **kwargs
):
    return _write_out(out, fused_infer_attention_score_v2(query, key, value, **kwargs))
