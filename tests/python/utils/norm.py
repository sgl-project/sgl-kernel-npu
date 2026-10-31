"""Finite input builders for fused normalization interface tests."""

from types import SimpleNamespace

import torch


def generate_mla_data(shape, dtype, bias_mode="none", weight_dtype=None, seed=17):
    """shape = (tokens, q_lora_rank, kv_lora_rank, qk_rope_dim)."""
    torch.manual_seed(seed)
    tokens, q_rank, kv_rank, rope_dim = shape
    weight_dtype = dtype if weight_dtype is None else weight_dtype
    norms = []
    for name, width in (("q", q_rank), ("k", kv_rank)):
        weight = (1.0 + torch.randn(width) * 0.125).to(weight_dtype)
        bias = (
            torch.randn(width).to(weight_dtype) * 0.125
            if bias_mode in (name, "both")
            else None
        )
        norms.append(SimpleNamespace(weight=weight, bias=bias))
    return {
        "fused_qkv_a_proj_out": torch.randn(
            tokens, q_rank + kv_rank + rope_dim, dtype=dtype
        ),
        "q_a_layernorm": norms[0],
        "kv_a_layernorm": norms[1],
        "q_lora_rank": q_rank,
        "kv_lora_rank": kv_rank,
        "qk_rope_dim": rope_dim,
        "eps": 1e-6,
    }


def generate_qkv_data(
    shape,
    dtype,
    *,
    cache_dtype=torch.float32,
    weight_dtype=None,
    norm=True,
    bias=False,
    gated=False,
    eps=1e-6,
    seed=17
):
    """shape = (tokens, local Q heads, local KV heads, head_dim, rope_dim).

    Build SGLang's duplicated-half [N, 1, 1, R] position tables. Token positions
    and weights are synthetic; the selected model widths come from each suite.
    """
    torch.manual_seed(seed)
    tokens, q_heads, kv_heads, dim, rope_dim = shape
    q_width, kv_width = q_heads * dim, kv_heads * dim
    weight_dtype = dtype if weight_dtype is None else weight_dtype
    positions = torch.arange(tokens, dtype=torch.float32) * 17
    inv_frequency = 10000.0 ** (
        -torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim
    )
    angles = positions[:, None] * inv_frequency[None, :]
    angles = angles.repeat(1, 2)
    case = {
        "input": torch.randn(
            tokens, q_width * (2 if gated else 1) + 2 * kv_width, dtype=dtype
        ),
        "sin": angles.sin().to(cache_dtype).reshape(tokens, 1, 1, rope_dim),
        "cos": angles.cos().to(cache_dtype).reshape(tokens, 1, 1, rope_dim),
        "q_hidden_size": q_width,
        "kv_hidden_size": kv_width,
        "head_dim": dim,
        "eps": eps if norm else None,
        "q_weight": (
            ((0.0 if gated else 1.0) + torch.randn(dim) * 0.125).to(weight_dtype)
            if norm
            else None
        ),
        "k_weight": (
            ((0.0 if gated else 1.0) + torch.randn(dim) * 0.125).to(weight_dtype)
            if norm
            else None
        ),
    }
    if gated:
        case["rope_dim"] = rope_dim
    else:
        case["q_bias"] = torch.randn(dim).to(weight_dtype) * 0.125 if bias else None
        case["k_bias"] = torch.randn(dim).to(weight_dtype) * 0.125 if bias else None
        case["is_neox_style"] = True
    return case


def generate_pos_cache_data(
    shape,
    dtype,
    *,
    cache_dtype=torch.float32,
    weight_dtype=None,
    norm=True,
    bias=False,
    cast_norm_to_bf16=True,
    max_seq=4096,
    seed=17
):
    """Build the [cos_half | sin_half] cache used by LLaDA2 and MiniMax M3."""
    case = generate_qkv_data(
        shape,
        dtype,
        cache_dtype=cache_dtype,
        weight_dtype=weight_dtype,
        norm=norm,
        bias=bias,
        seed=seed,
    )
    case["input_tensor"] = case.pop("input")
    del case["sin"], case["cos"], case["is_neox_style"]
    tokens, _, _, _, rope_dim = shape
    inv_frequency = 10000.0 ** (
        -torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim
    )
    angles = (
        torch.arange(max_seq, dtype=torch.float32)[:, None] * inv_frequency[None, :]
    )
    case.update(
        positions=(torch.arange(tokens, dtype=torch.int64) * 17) % max_seq,
        cos_sin_cache=torch.cat((angles.cos(), angles.sin()), dim=-1).to(cache_dtype),
        rope_dim=rope_dim,
        cast_norm_to_bf16=cast_norm_to_bf16,
    )
    return case
