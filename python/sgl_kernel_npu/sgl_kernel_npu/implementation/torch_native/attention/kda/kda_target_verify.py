from typing import Optional

import torch
import torch.nn.functional as F


def kda_target_verify(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    scale: Optional[float] = None,
    gates_are_preactivated: Optional[bool] = None,
) -> torch.Tensor:
    """Evaluate normalized KDA recurrence and write selected state snapshots."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [1, tokens, heads, dim]")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("the leading q, k, and v dimension must be one")
    if cache_steps <= 0 or q.shape[1] % cache_steps != 0:
        raise ValueError("tokens must be divisible by positive cache_steps")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q, k, and v token dimensions must match")

    batch = q.shape[1] // cache_steps
    h_q, key_dim = q.shape[2:]
    h_k = k.shape[2]
    h_v, value_dim = v.shape[2:]
    a_has_leading_singleton = a.ndim == 4
    b_has_leading_singleton = b.ndim == 3
    if a_has_leading_singleton != b_has_leading_singleton:
        raise ValueError("a and b must use the leading singleton together")
    if gates_are_preactivated is None:
        gates_are_preactivated = a_has_leading_singleton
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError("4D a must have a leading singleton dimension")
        a = a.squeeze(0)
    if b.ndim == 3:
        if b.shape[0] != 1:
            raise ValueError("3D b must have a leading singleton dimension")
        b = b.squeeze(0)
    if k.shape[3] != key_dim:
        raise ValueError("q and k key dimensions must match")
    if h_v % h_q != 0 or h_v % h_k != 0:
        raise ValueError("value heads must be divisible by q and k heads")
    if tuple(a.shape) != (q.shape[1], h_k, key_dim):
        raise ValueError("a must have shape [tokens, H_k, K]")
    if tuple(b.shape) != (q.shape[1], h_v):
        raise ValueError("b must have shape [tokens, H_v]")
    if not gates_are_preactivated and (
        A_log.numel() != h_k or tuple(dt_bias.shape) != (h_k, key_dim)
    ):
        raise ValueError("A_log and dt_bias shapes do not match KDA heads")
    if initial_state_source.ndim != 4 or tuple(initial_state_source.shape[1:]) != (
        h_v,
        value_dim,
        key_dim,
    ):
        raise ValueError("initial state must have shape [pool, H_v, V, K]")
    if intermediate_states_buffer.ndim != 5 or tuple(
        intermediate_states_buffer.shape[1:]
    ) != (cache_steps, h_v, value_dim, key_dim):
        raise ValueError("intermediate state must have shape [scratch, T, H_v, V, K]")
    if initial_state_indices.ndim != 1 or initial_state_indices.numel() < batch:
        raise ValueError("initial_state_indices must contain at least B entries")
    if (
        intermediate_state_indices.ndim != 1
        or intermediate_state_indices.numel() < batch
    ):
        raise ValueError("intermediate_state_indices must contain at least B entries")

    # SGLang produces q/k/v as views of a packed QKV tensor. The kernel consumes
    # explicit strides so serving can avoid five per-layer materializations.
    tensors = [
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        intermediate_states_buffer,
        intermediate_state_indices,
    ]
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    initial_state_indices = initial_state_indices.contiguous()
    intermediate_state_indices = intermediate_state_indices.contiguous()
    if initial_state_source.dtype != intermediate_states_buffer.dtype:
        raise ValueError("persistent and intermediate state dtypes must match")
    if initial_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("initial_state_indices must be int32 or int64")
    if intermediate_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("intermediate_state_indices must be int32 or int64")

    if scale is None:
        scale = key_dim**-0.5
    if scale <= 0:
        raise ValueError("scale must be positive")
    if key_dim > 256:
        raise ValueError("key dimensions greater than 256 are unsupported")
    qf, kf, vf = q[0].float(), k[0].float(), v[0].float()
    if key_dim == 128 and value_dim == 128:
        qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-12)
        kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-12)
    else:
        qf = qf / (qf.square().sum(-1, keepdim=True).sqrt() + 1e-6)
        kf = kf / (kf.square().sum(-1, keepdim=True).sqrt() + 1e-6)
    qf = qf.repeat_interleave(h_v // h_q, dim=1) * scale
    kf = kf.repeat_interleave(h_v // h_k, dim=1)
    if gates_are_preactivated:
        log_decay, beta = a.float(), b.float()
    else:
        log_decay = -A_log.float().reshape(h_k, 1).exp() * F.softplus(
            a.float() + dt_bias.float()
        )
        beta = b.float().sigmoid()
    log_decay = log_decay.repeat_interleave(h_v // h_k, dim=1)
    out = torch.empty_like(v)
    for request in range(batch):
        source = int(initial_state_indices[request])
        destination = int(intermediate_state_indices[request])
        state = (
            initial_state_source[source].float().clone()
            if source >= 0
            else torch.zeros((h_v, value_dim, key_dim), device=q.device)
        )
        for step in range(cache_steps):
            token = request * cache_steps + step
            state = state * log_decay[token].exp().unsqueeze(-2)
            predicted = torch.einsum("hvk,hk->hv", state, kf[token])
            delta = (vf[token] - predicted) * beta[token].unsqueeze(-1)
            state = state + delta.unsqueeze(-1) * kf[token].unsqueeze(-2)
            out[0, token] = torch.einsum("hvk,hk->hv", state, qf[token])
            if destination >= 0:
                intermediate_states_buffer[destination, step].copy_(state)
    return out
