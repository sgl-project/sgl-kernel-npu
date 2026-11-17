"""PyTorch tensor-parallel QKV RMSNorm and RoPE implementation."""

import torch
import torch.distributed as dist


def compute_local(input_q, input_k, input_v, q, k, v, qk_var):
    q.copy_(input_q)
    k.copy_(input_k)
    v.copy_(input_v)
    qk_var[:, 0].copy_(torch.mean(input_q.float() ** 2, dim=-1))
    qk_var[:, 1].copy_(torch.mean(input_k.float() ** 2, dim=-1))


def _normalize_rotate(values, variance, weight, cos, sin, eps, inv_tp_world):
    scale = 1.0 / torch.sqrt(variance * inv_tp_world + eps)
    # Store normalized Q/K in input precision,
    # then reloads them before applying RoPE.
    normalized = (
        (values.float() * scale[:, None, None] * weight.float())
        .to(values.dtype)
        .float()
    )
    half = cos.shape[-1]
    left, right = normalized[..., :half], normalized[..., half : 2 * half]
    rotated = torch.cat(
        (
            left * cos.float() - right * sin.float(),
            right * cos.float() + left * sin.float(),
            normalized[..., 2 * half :],
        ),
        dim=-1,
    )
    values.copy_(rotated)


def apply_global(q, k, qk_var, q_weight, k_weight, cos, sin, eps, inv_tp_world):
    _normalize_rotate(q, qk_var[:, 0], q_weight, cos, sin, eps, inv_tp_world)
    _normalize_rotate(k, qk_var[:, 1], k_weight, cos, sin, eps, inv_tp_world)


def split_qkv_tp_rmsnorm_rope(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_dim: int,
    tp_world: int,
    tp_group: torch.distributed.ProcessGroup,
):
    assert head_dim == 1 << (head_dim - 1).bit_length()
    assert q_hidden_size % kv_hidden_size == 0
    batch = input.shape[0]
    q = torch.empty((batch, q_hidden_size), device=input.device, dtype=input.dtype)
    k = torch.empty((batch, kv_hidden_size), device=input.device, dtype=input.dtype)
    v = torch.empty_like(k)
    if batch == 0:
        return q, k, v
    qk_var = torch.empty((batch, 2), device=input.device, dtype=torch.float32)
    compute_local(
        input[:, :q_hidden_size],
        input[:, q_hidden_size : q_hidden_size + kv_hidden_size],
        input[:, q_hidden_size + kv_hidden_size :],
        q,
        k,
        v,
        qk_var,
    )
    if tp_world > 1:
        dist.all_reduce(qk_var, group=tp_group)
    q_heads = q.view(batch, q_hidden_size // head_dim, head_dim)
    k_heads = k.view(batch, kv_hidden_size // head_dim, head_dim)
    cos_half = cos.reshape(batch, rotary_dim)[:, : rotary_dim // 2].unsqueeze(1)
    sin_half = sin.reshape(batch, rotary_dim)[:, : rotary_dim // 2].unsqueeze(1)
    apply_global(
        q_heads,
        k_heads,
        qk_var,
        q_weight.reshape(-1)[:q_hidden_size].view(q_hidden_size // head_dim, head_dim),
        k_weight.reshape(-1)[:kv_hidden_size].view(
            kv_hidden_size // head_dim, head_dim
        ),
        cos_half,
        sin_half,
        eps,
        1.0 / tp_world,
    )
    return q, k, v
