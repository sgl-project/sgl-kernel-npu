import torch

from ._qkv_compute import flatten_heads, normalize


def _normalize_rope(values, sin, cos, eps, weight, bias, cast_norm_to_bf16):
    values = normalize(values, eps, weight, bias)
    if cast_norm_to_bf16:
        values = values.to(torch.bfloat16).float()
    half = sin.shape[-1]
    left, right = values[..., :half], values[..., half : 2 * half]
    # The native kernel rounds the RoPE result and pass-through tail to BF16
    # even when cast_norm_to_bf16=False or the input/output dtype is FP32/FP16.
    roped = torch.cat(
        (
            left * cos.float() - right * sin.float(),
            right * cos.float() + left * sin.float(),
        ),
        dim=-1,
    )
    return torch.cat(
        (roped.to(torch.bfloat16), values[..., 2 * half :].to(torch.bfloat16)), dim=-1
    )


def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    q, k, v, sin, cos, eps, q_weight, k_weight, q_bias, k_bias, cast_norm_to_bf16, dtype
):
    q = _normalize_rope(q, sin, cos, eps, q_weight, q_bias, cast_norm_to_bf16)
    k = _normalize_rope(k, sin, cos, eps, k_weight, k_bias, cast_norm_to_bf16)
    return flatten_heads(q, dtype), flatten_heads(k, dtype), v.clone().contiguous()
