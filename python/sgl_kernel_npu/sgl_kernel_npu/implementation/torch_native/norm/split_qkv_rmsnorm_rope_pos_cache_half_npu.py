"""Position-cache RoPE with the fused kernel's explicit BF16 rounding policy."""

import torch

from .norm_utils import rms_norm, rotary, validate_pos_cache


def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    input_tensor,
    positions,
    cos_sin_cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    rope_dim=None,
    cast_norm_to_bf16=True,
    wide_grid_min_tokens=1024,
):
    rope_dim = validate_pos_cache(
        input_tensor,
        positions,
        cos_sin_cache,
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
    )
    tokens = input_tensor.shape[0]
    # Match the device kernel's int32 conversion before clamping. All index
    # handling stays on-device; no min/max/item() host synchronization.
    positions = (
        positions.reshape(-1)
        .to(torch.int32)
        .long()
        .clamp(0, cos_sin_cache.shape[0] - 1)
    )
    selected = cos_sin_cache.index_select(0, positions).float()
    cos, sin = selected.split(rope_dim // 2, dim=-1)
    cos = cos.repeat(1, 2)
    sin = sin.repeat(1, 2)
    q, k, v = input_tensor.split(
        (q_hidden_size, kv_hidden_size, kv_hidden_size), dim=-1
    )
    outputs = []
    for values, width, weight, bias in (
        (q, q_hidden_size, q_weight, q_bias),
        (k, kv_hidden_size, k_weight, k_bias),
    ):
        values = values.float().reshape(tokens, width // head_dim, head_dim)
        if eps is not None:
            values = rms_norm(values, weight, eps, bias)
        if cast_norm_to_bf16:
            values = values.to(torch.bfloat16).float()
        # The existing kernel rounds both rotated and pass-through Q/K values
        # to BF16, even when the input/output dtype is FP16 or FP32.
        result = rotary(values, sin, cos, rope_dim).to(torch.bfloat16)
        outputs.append(result.to(input_tensor.dtype).reshape(tokens, width))
    return outputs[0], outputs[1], v.clone().contiguous()
