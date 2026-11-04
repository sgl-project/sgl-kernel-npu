"""TP QKV computations around the operator-owned collective boundary."""

# PyTorch replaces NumPy RoPE so computation stays on the input device;
# dtype rounding and the unrotated suffix support the public API's additional cases.
import torch


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
