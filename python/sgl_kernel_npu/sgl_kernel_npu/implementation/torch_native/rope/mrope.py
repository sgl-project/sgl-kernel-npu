"""PyTorch reference for standard RoPE and segmented MRoPE."""

import torch

from .rotary_mul import rotary_mul


def _apply_rope(states, cos, sin, head_size, rotary_mode):
    rotary_dim = cos.size(-1)
    heads = states.reshape(states.size(0), states.size(1) // head_size, head_size)
    rotated = rotary_mul(heads[..., :rotary_dim], cos, sin, rotary_mode)
    return torch.cat((rotated, heads[..., rotary_dim:]), dim=-1).reshape_as(states)


def mrope(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    *,
    mrope_section=None,
    rotary_mode="half",
    cache_mode="default",
):
    if cache_mode != "default":
        raise ValueError("torch_native mrope supports cache_mode='default'")

    cos, sin = cos_sin_cache[positions.long()].chunk(2, dim=-1)
    if positions.ndim == 2:
        if (
            mrope_section is None
            or len(mrope_section) != positions.size(0)
            or sum(mrope_section) != cos.size(-1)
        ):
            raise ValueError(
                "mrope_section must partition the cache frequencies by axis"
            )
        cos = torch.cat(
            [part[i] for i, part in enumerate(cos.split(mrope_section, -1))], -1
        )
        sin = torch.cat(
            [part[i] for i, part in enumerate(sin.split(mrope_section, -1))], -1
        )

    if rotary_mode == "half":
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    elif rotary_mode in ("interleave", "interleaved"):
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        rotary_mode = "interleave"
    else:
        raise ValueError("torch_native mrope supports half and interleave")

    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    return (
        _apply_rope(query, cos, sin, head_size, rotary_mode),
        _apply_rope(key, cos, sin, head_size, rotary_mode),
    )
