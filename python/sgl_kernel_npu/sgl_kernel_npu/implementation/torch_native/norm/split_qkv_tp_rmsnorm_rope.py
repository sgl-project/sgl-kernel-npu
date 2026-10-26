"""TP QKV computations around the operator-owned collective boundary."""

import torch


def _chunked_mean_square(values, block):
    values = values.float()
    total = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
    compensation = torch.zeros_like(total)
    for start in range(0, values.shape[1], block):
        chunk = values[:, start : start + block]
        term = chunk.square().sum(-1) - compensation
        updated = total + term
        compensation = (updated - total) - term
        total = updated
    return total / values.shape[1]


def compute_local(input_q, input_k, input_v, q, k, v, qk_var, q_block, kv_block):
    q.copy_(input_q)
    k.copy_(input_k)
    v.copy_(input_v)
    qk_var[:, 0].copy_(_chunked_mean_square(input_q, q_block))
    qk_var[:, 1].copy_(_chunked_mean_square(input_k, kv_block))


def _normalize_rotate(values, variance, weight, cos, sin, eps, inv_tp_world):
    scale = 1.0 / torch.sqrt(variance * inv_tp_world + eps)
    # The original second stage stores normalized Q/K in input precision,
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
