"""Shared input generation for public norm accuracy/performance tests."""

import torch


def seed_inputs(seed=42):
    torch.npu.manual_seed(seed)


def make_qkv_inputs(
    dtype,
    *,
    rows=3,
    q_heads=4,
    kv_heads=2,
    head_dim=64,
    rope_dim=None,
    eps=1e-5,
    bias=False,
    is_neox_style=True,
    cache_dtype=torch.float32,
):
    """Use representative head layouts; these are not captured production calls."""
    seed_inputs()
    rope_dim = head_dim if rope_dim is None else rope_dim
    q_width, kv_width = q_heads * head_dim, kv_heads * head_dim
    width = q_width + 2 * kv_width
    x = torch.randn(rows, width, dtype=dtype)
    frequencies = 10000.0 ** (-torch.arange(0, rope_dim, 2).float() / rope_dim)
    angles = torch.arange(rows).float()[:, None] * frequencies[None, :]
    full_angles = torch.cat((angles, angles), dim=-1)
    return {
        "input": x,
        "sin": full_angles.sin().to(cache_dtype),
        "cos": full_angles.cos().to(cache_dtype),
        "q_hidden_size": q_width,
        "kv_hidden_size": kv_width,
        "head_dim": head_dim,
        "eps": eps,
        "q_weight": (0.5 + torch.rand(head_dim)).to(dtype) if eps is not None else None,
        "k_weight": (0.5 + torch.rand(head_dim)).to(dtype) if eps is not None else None,
        "q_bias": (0.1 * torch.randn(head_dim)).to(dtype) if bias else None,
        "k_bias": (0.1 * torch.randn(head_dim)).to(dtype) if bias else None,
        "is_neox_style": is_neox_style,
    }


def make_position_cache(rows, rope_dim, dtype=torch.float32):
    frequencies = 10000.0 ** (-torch.arange(0, rope_dim, 2).float() / rope_dim)
    angles = torch.arange(rows).float()[:, None] * frequencies[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)


def make_tp_inputs(
    dtype, *, rows=3, q_width=512, kv_width=128, head_dim=64, rotary_dim=64, seed=42
):
    seed_inputs(seed)
    cache = make_position_cache(max(rows, 1), rotary_dim, torch.float32)[:rows]
    cos_half, sin_half = cache.chunk(2, dim=-1)
    return {
        "input": torch.randn(rows, q_width + 2 * kv_width, dtype=dtype),
        "cos": torch.cat((cos_half, cos_half), dim=-1),
        "sin": torch.cat((sin_half, sin_half), dim=-1),
        "q_hidden_size": q_width,
        "kv_hidden_size": kv_width,
        "head_dim": head_dim,
        "eps": 1e-6,
        "q_weight": (0.5 + torch.rand(q_width)).to(dtype),
        "k_weight": (0.5 + torch.rand(kv_width)).to(dtype),
        "rotary_dim": rotary_dim,
        "tp_world": 1,
        "tp_group": None,
    }
