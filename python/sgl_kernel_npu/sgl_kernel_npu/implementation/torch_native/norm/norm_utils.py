"""Independent PyTorch normalization checks and FP32 mathematical helpers."""

import math

import torch


def positive_int(value, name):
    assert (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    ), f"{name} must be a positive Python int, got {value!r}"


def epsilon(value):
    assert isinstance(value, (int, float)) and not isinstance(
        value, bool
    ), f"eps must be a finite positive Python number, got {value!r}"
    assert (
        math.isfinite(value) and value > 0
    ), f"eps must be finite and positive, got {value!r}"


def floating_tensor(tensor, name, device=None, contiguous=True):
    assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor"
    assert tensor.layout == torch.strided, f"{name} must have strided layout"
    assert tensor.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"{name} must have float16, bfloat16 or float32 dtype, got {tensor.dtype}"
    if device is not None:
        assert (
            tensor.device == device
        ), f"{name} must be on {device}, got {tensor.device}"
    if contiguous:
        assert (
            tensor.is_contiguous()
        ), f"{name} must be contiguous, got stride={tensor.stride()}"


def vector(tensor, name, size, device):
    floating_tensor(tensor, name, device)
    assert (
        tensor.numel() >= size
    ), f"{name} must have at least {size} values, got {tensor.numel()}"


def validate_split_qk_norm(input, q_norm, k_norm, q_rank, kv_rank, rope_dim, eps):
    floating_tensor(input, "fused_qkv_a_proj_out")
    for name, value in (
        ("q_lora_rank", q_rank),
        ("kv_lora_rank", kv_rank),
        ("qk_rope_dim", rope_dim),
    ):
        positive_int(value, name)
    assert input.ndim == 2, f"fused_qkv_a_proj_out must be 2D, got {tuple(input.shape)}"
    assert (
        input.shape[1] == q_rank + kv_rank + rope_dim
    ), f"input width must equal q_lora_rank + kv_lora_rank + qk_rope_dim, got {input.shape[1]}"
    epsilon(eps)
    for name, norm, size in (
        ("q_a_layernorm", q_norm, q_rank),
        ("kv_a_layernorm", k_norm, kv_rank),
    ):
        assert hasattr(norm, "weight"), f"{name} must expose a weight tensor"
        vector(norm.weight, f"{name}.weight", size, input.device)
        bias = getattr(norm, "bias", None)
        if bias is not None:
            vector(bias, f"{name}.bias", size, input.device)


def qkv_shape(input, q_hidden_size, kv_hidden_size, head_dim, gated=False):
    floating_tensor(input, "input")
    assert input.ndim == 2, f"input must be 2D, got {tuple(input.shape)}"
    for name, value in (
        ("q_hidden_size", q_hidden_size),
        ("kv_hidden_size", kv_hidden_size),
        ("head_dim", head_dim),
    ):
        positive_int(value, name)
    assert (
        kv_hidden_size % head_dim == 0
    ), "kv_hidden_size must be divisible by head_dim"
    assert (
        q_hidden_size % kv_hidden_size == 0
    ), "q_hidden_size must be divisible by kv_hidden_size"
    width = q_hidden_size * (2 if gated else 1) + 2 * kv_hidden_size
    assert input.shape[1] == width, f"input width must be {width}, got {input.shape[1]}"


def affine(input, head_dim, eps, q_weight, k_weight, q_bias=None, k_bias=None):
    if eps is not None:
        epsilon(eps)
        vector(q_weight, "q_weight", head_dim, input.device)
        vector(k_weight, "k_weight", head_dim, input.device)
    assert (q_bias is None) == (
        k_bias is None
    ), "q_bias and k_bias must be supplied together"
    if q_bias is not None:
        vector(q_bias, "q_bias", head_dim, input.device)
        vector(k_bias, "k_bias", head_dim, input.device)


def trig_tables(input, sin, cos, head_dim):
    floating_tensor(sin, "sin", input.device)
    floating_tensor(cos, "cos", input.device)
    assert (
        sin.ndim >= 2 and cos.ndim >= 2
    ), "sin and cos must have a token and rotary dimension"
    rope_dim = sin.shape[-1]
    positive_int(rope_dim, "rope_dim")
    assert (
        rope_dim % 2 == 0 and rope_dim <= head_dim
    ), "rope_dim must be even and <= head_dim"
    assert cos.shape[-1] == rope_dim, "sin and cos must have the same rotary dimension"
    count = input.shape[0] * rope_dim
    assert (
        sin.numel() == count and cos.numel() == count
    ), "sin and cos must contain exactly tokens * rope_dim values"
    return rope_dim


def validate_split_qkv(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps,
    q_weight,
    k_weight,
    q_bias,
    k_bias,
    is_neox_style,
):
    qkv_shape(input, q_hidden_size, kv_hidden_size, head_dim)
    trig_tables(input, sin, cos, head_dim)
    affine(input, head_dim, eps, q_weight, k_weight, q_bias, k_bias)
    assert isinstance(is_neox_style, bool), "is_neox_style must be bool"


def validate_split_qkvgate(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
):
    qkv_shape(input, q_hidden_size, kv_hidden_size, head_dim, gated=True)
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of two"
    positive_int(rope_dim, "rope_dim")
    assert (
        trig_tables(input, sin, cos, head_dim) == rope_dim
    ), "rope_dim must match the sin/cos width"
    epsilon(eps)
    affine(input, head_dim, eps, q_weight, k_weight)


def validate_pos_cache(
    input,
    positions,
    cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps,
    q_weight,
    k_weight,
    q_bias,
    k_bias,
    rope_dim,
    cast_norm_to_bf16,
    wide_grid_min_tokens,
):
    qkv_shape(input, q_hidden_size, kv_hidden_size, head_dim)
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of two"
    rope_dim = head_dim if rope_dim is None else rope_dim
    positive_int(rope_dim, "rope_dim")
    assert (
        rope_dim % 2 == 0 and rope_dim <= head_dim
    ), "rope_dim must be even and <= head_dim"
    affine(input, head_dim, eps, q_weight, k_weight, q_bias, k_bias)
    assert isinstance(positions, torch.Tensor), "positions must be a torch.Tensor"
    assert positions.layout == torch.strided, "positions must have strided layout"
    assert positions.dtype in (
        torch.int32,
        torch.int64,
    ), "positions must have int32 or int64 dtype"
    assert positions.device == input.device, "positions must be on the input device"
    assert (
        positions.numel() == input.shape[0]
    ), "positions must contain one index per token"
    floating_tensor(cache, "cos_sin_cache", input.device, contiguous=False)
    assert (
        cache.ndim == 2 and cache.shape[0] > 0 and cache.shape[1] == rope_dim
    ), "cos_sin_cache must have shape [positive max_seq, rope_dim]"
    assert isinstance(cast_norm_to_bf16, bool), "cast_norm_to_bf16 must be bool"
    positive_int(wide_grid_min_tokens, "wide_grid_min_tokens")
    return rope_dim


def rms_norm(values, weight, eps, bias=None, gemma=False):
    values = values.float()
    width = values.shape[-1]
    weight = weight.reshape(-1)[:width].float()
    factor = weight + 1.0 if gemma else weight
    result = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    result = result * factor
    if bias is not None:
        result = result + bias.reshape(-1)[:width].float()
    return result


def rotary(values, sin, cos, rope_dim, is_neox_style=True):
    tokens = values.shape[0]
    values = values.float()
    sin = sin.float().reshape(tokens, 1, rope_dim)
    cos = cos.float().reshape(tokens, 1, rope_dim)
    rotated = values[..., :rope_dim]
    half = rope_dim // 2
    if is_neox_style:
        paired = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
    else:
        # The interface receives duplicated-half tables, as SGLang constructs
        # them; adjacent-pair rotation uses only the first half of each table.
        sin = sin[..., :half].repeat_interleave(2, dim=-1)
        cos = cos[..., :half].repeat_interleave(2, dim=-1)
        paired = torch.stack((-rotated[..., 1::2], rotated[..., ::2]), dim=-1).flatten(
            -2
        )
    result = rotated * cos + paired * sin
    return torch.cat((result, values[..., rope_dim:]), dim=-1)
