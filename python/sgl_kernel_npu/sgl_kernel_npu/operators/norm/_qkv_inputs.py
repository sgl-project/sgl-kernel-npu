"""Metadata checks and input preparation for the public QKV adapters.

Checks use tensor metadata only; the position-cache path keeps clamping on device
so preparing a Triton call does not introduce a host synchronization into graphs.
"""

import torch


def _validate_layout(input, q_hidden_size, kv_hidden_size, head_dim, *, gemma=False):
    if input.ndim != 2:
        raise ValueError("input must be a 2-D concatenated QKV tensor")
    if head_dim <= 0 or q_hidden_size <= 0 or kv_hidden_size <= 0:
        raise ValueError("head_dim and Q/KV widths must be positive")
    if (
        q_hidden_size % kv_hidden_size
        or q_hidden_size % head_dim
        or kv_hidden_size % head_dim
    ):
        raise ValueError(
            "Q/KV widths must contain complete heads and Q must be a multiple of KV"
        )
    expected = (2 if gemma else 1) * q_hidden_size + 2 * kv_hidden_size
    if input.shape[1] != expected:
        raise ValueError("input width does not match the concatenated Q/K/V layout")


def _validate_weights(input, head_dim, eps, q_weight, k_weight, q_bias, k_bias):
    if eps is not None:
        for name, value in (("q_weight", q_weight), ("k_weight", k_weight)):
            if value is None or value.numel() < head_dim:
                raise ValueError(
                    f"{name} must have at least head_dim={head_dim} elements when eps is set"
                )
            if value.device != input.device:
                raise ValueError(f"{name} must be on the input device")
    # The original kernel uses q_bias to enable both Q and K bias loads.
    if q_bias is not None:
        for name, value in (("q_bias", q_bias), ("k_bias", k_bias)):
            if value is None or value.numel() < head_dim:
                raise ValueError(
                    f"{name} must have at least head_dim={head_dim} elements when q_bias is set"
                )
            if value.device != input.device:
                raise ValueError(f"{name} must be on the input device")


def prepare_qkv(
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
    *,
    rope_dim=None,
    gemma=False,
):
    _validate_layout(input, q_hidden_size, kv_hidden_size, head_dim, gemma=gemma)
    if sin.ndim == 0 or cos.ndim == 0:
        raise ValueError("sin and cos must have a rotary dimension")
    rope_dim = sin.shape[-1] if rope_dim is None else rope_dim
    if rope_dim < 0 or rope_dim % 2 or rope_dim > head_dim:
        raise ValueError("rope_dim must be even and within head_dim")
    if (
        sin.shape[-1] != rope_dim
        or cos.shape[-1] != rope_dim
        or sin.numel() != input.shape[0] * rope_dim
        or cos.numel() != input.shape[0] * rope_dim
    ):
        raise ValueError("sin/cos must contain one rope_dim row per input token")
    if sin.device != input.device or cos.device != input.device:
        raise ValueError("sin/cos must be on the input device")
    if gemma:
        if head_dim & (head_dim - 1):
            raise ValueError("Gemma head_dim must be a power of two")
        if eps is None or q_weight is None or k_weight is None:
            raise ValueError("Gemma normalization requires eps and both weights")
    _validate_weights(input, head_dim, eps, q_weight, k_weight, q_bias, k_bias)
    # Native kernels use flat pointer offsets, not arbitrary tensor strides.
    return tuple(
        v.contiguous() if v is not None else None
        for v in (input, sin, cos, q_weight, k_weight, q_bias, k_bias)
    )


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
    _validate_layout(input_tensor, q_hidden_size, kv_hidden_size, head_dim)
    rope_dim = head_dim if rope_dim is None else rope_dim
    if head_dim & (head_dim - 1) or rope_dim < 0 or rope_dim % 2 or rope_dim > head_dim:
        raise ValueError(
            "head_dim must be a power of two and rope_dim must be even within it"
        )
    if (
        positions.numel() != input_tensor.shape[0]
        or positions.device != input_tensor.device
    ):
        raise ValueError(
            "positions must contain one index per token on the input device"
        )
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] < 1
        or cos_sin_cache.shape[1] < rope_dim
        or cos_sin_cache.device != input_tensor.device
    ):
        raise ValueError(
            "cos_sin_cache must have nonempty rows and at least rope_dim columns on the input device"
        )
    _validate_weights(input_tensor, head_dim, eps, q_weight, k_weight, q_bias, k_bias)
    if positions.dtype not in (torch.int32, torch.int64):
        positions = positions.to(torch.int32)
    return positions.contiguous(), cos_sin_cache.contiguous(), rope_dim


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
