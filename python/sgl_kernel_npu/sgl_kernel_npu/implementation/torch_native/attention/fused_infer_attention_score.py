"""Independent dense, unquantized FIA reference shared by both SDK schemas."""

import torch


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
    atten_mask=None,
    pse_shift=None,
    sparse_mode=0,
    inner_precise=0,
    softmax_lse_flag=False,
    pre_tokens=2147483647,
    next_tokens=2147483647,
):
    """Evaluate dense BNSD, BSND or BSH attention using explicit masks only."""
    if (
        sparse_mode not in (0, 1)
        or inner_precise not in (0, 1, 2, 3)
        or (pre_tokens, next_tokens) != (2147483647, 2147483647)
    ):
        raise NotImplementedError("torch_native requires dense FIA")
    if query.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != query.dtype for tensor in (key, value)
    ):
        raise NotImplementedError("torch_native requires unquantized FP16/BF16 Q/K/V")
    q, k, v = (tensor.detach().cpu() for tensor in (query, key, value))
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
