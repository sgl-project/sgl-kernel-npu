"""CPU reference for split QKV, tensor-parallel RMSNorm and NeoX RoPE."""

import torch
import torch.distributed as dist

from ...torch_native.norm.split_qkv_tp_rmsnorm_rope import apply_global, compute_local


def split_qkv_tp_rmsnorm_rope(
    input,
    cos,
    sin,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps,
    q_weight,
    k_weight,
    rotary_dim,
    tp_world,
    tp_group,
):
    assert head_dim == 1 << (head_dim - 1).bit_length()
    assert q_hidden_size % kv_hidden_size == 0
    device = input.device
    parts = input.cpu().split((q_hidden_size, kv_hidden_size, kv_hidden_size), dim=1)
    if input.shape[0] == 0:
        return tuple(part.clone().to(device) for part in parts)

    q, k, v = (part.new_empty(part.shape) for part in parts)
    means = q.new_empty((input.shape[0], 2), dtype=torch.float32)
    compute_local(*parts, q, k, v, means)
    if tp_world > 1:
        means = means.to(device)
        dist.all_reduce(means, group=tp_group)
        means = means.cpu() / tp_world

    half = rotary_dim // 2
    apply_global(
        q.view(input.shape[0], -1, head_dim),
        k.view(input.shape[0], -1, head_dim),
        means,
        q_weight.cpu().flatten()[:q_hidden_size].view(-1, head_dim),
        k_weight.cpu().flatten()[:kv_hidden_size].view(-1, head_dim),
        cos.cpu().reshape(-1, rotary_dim)[:, None, :half],
        sin.cpu().reshape(-1, rotary_dim)[:, None, :half],
        eps,
        1.0,
    )
    return tuple(part.to(device) for part in (q, k, v))
