"""Reference tensor views and the original position-cache input preparation."""

import torch


def prepare_position_cache(
    input_tensor,
    positions,
    cos_sin_cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
    q_bias,
    k_bias,
):
    assert input_tensor.dim() == 2
    B, total_hidden = input_tensor.shape
    if rope_dim is None:
        rope_dim = head_dim
    assert rope_dim % 2 == 0 and rope_dim <= head_dim
    expected_total = q_hidden_size + 2 * kv_hidden_size
    assert total_hidden == expected_total
    pos = positions
    assert pos.numel() == B, f"positions must be [B], got numel={pos.numel()} B={B}"
    if pos.dtype not in (torch.int32, torch.int64):
        pos = pos.to(torch.int32)
    pos = pos.contiguous()
    cache = cos_sin_cache.contiguous()
    max_seq = cache.shape[0]
    assert max_seq >= 1, "cos_sin_cache must be non-empty"
    assert (
        1 << (head_dim - 1).bit_length()
    ) == head_dim, "this kernel assumes head_dim is power-of-2"
    assert q_hidden_size % kv_hidden_size == 0
    bias = q_bias is not None
    norms = eps is not None
    if norms:
        if q_weight is None or q_weight.numel() < head_dim:
            raise ValueError(
                f"When using RMSNorm (eps is not None), q_weight must have at least head_dim={head_dim} elements, "
                f"got {q_weight.numel() if q_weight is not None else 0}."
            )
        if k_weight is None or k_weight.numel() < head_dim:
            raise ValueError(
                f"When using RMSNorm (eps is not None), k_weight must have at least head_dim={head_dim} elements, "
                f"got {k_weight.numel() if k_weight is not None else 0}."
            )
    if bias:
        if q_bias is None or q_bias.numel() < head_dim:
            raise ValueError(
                f"When using bias (q_bias provided), q_bias must have at least head_dim={head_dim} elements, "
                f"got {q_bias.numel() if q_bias is not None else 0}."
            )
        if k_bias is None or k_bias.numel() < head_dim:
            raise ValueError(
                f"When using bias (k_bias provided), k_bias must have at least head_dim={head_dim} elements, "
                f"got {k_bias.numel() if k_bias is not None else 0}."
            )
    return pos, cache, rope_dim


def split_qkv_inputs(input, q_hidden_size, kv_hidden_size, head_dim):
    batch = input.shape[0]
    q, k, v = input.split((q_hidden_size, kv_hidden_size, kv_hidden_size), dim=-1)
    return (
        q.reshape(batch, q_hidden_size // head_dim, head_dim),
        k.reshape(batch, kv_hidden_size // head_dim, head_dim),
        v,
    )


def reference_weights(head_dim, eps, q_weight, k_weight, q_bias, k_bias):
    # Biases are loaded when q_bias is set, but affect output only with RMSNorm.
    return (
        q_weight.reshape(-1)[:head_dim] if eps is not None else None,
        k_weight.reshape(-1)[:head_dim] if eps is not None else None,
        (
            q_bias.reshape(-1)[:head_dim]
            if eps is not None and q_bias is not None
            else None
        ),
        (
            k_bias.reshape(-1)[:head_dim]
            if eps is not None and q_bias is not None
            else None
        ),
    )
