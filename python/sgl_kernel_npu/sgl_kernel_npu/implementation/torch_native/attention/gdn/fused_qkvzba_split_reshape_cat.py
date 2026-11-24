"""PyTorch references for the QKVZ/BA split-reshape-cat operators."""

import torch


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
):
    """Reorder per-head-group interleaved QKVZ/BA rows into QKV, z, b and a."""
    B = mixed_qkvz.shape[0]
    v_group_size = num_heads_v // num_heads_qk  # = 2

    # Step 1: Reshape to [B, num_heads_qk, per_head_dim]
    per_head_dim = 2 * head_qk + 2 * v_group_size * head_v  # 768
    x = mixed_qkvz.view(B, num_heads_qk, per_head_dim)

    # Extract components per head
    q = x[:, :, :head_qk]  # [B, 4, 128]
    k = x[:, :, head_qk : 2 * head_qk]  # [B, 4, 128]
    v_groups = x[:, :, 2 * head_qk : 2 * head_qk + v_group_size * head_v]  # [B, 4, 256]
    z_groups = x[:, :, 2 * head_qk + v_group_size * head_v :]  # [B, 4, 256]

    # Reshape V and Z to [B, num_heads_v, head_v]
    v = v_groups.reshape(B, num_heads_v, head_v)  # [B, 8, 128]
    z = z_groups.reshape(B, num_heads_v, head_v)  # [B, 8, 128]

    # Build mixed_qkv = [Q_flat, K_flat, V_flat]
    # Q_flat: concatenate all q heads → [B, 4*128]
    q_flat = q.reshape(B, -1)
    k_flat = k.reshape(B, -1)
    v_flat = v.reshape(B, -1)
    mixed_qkv = torch.cat([q_flat, k_flat, v_flat], dim=1)  # [B, 2048]

    # Process mixed_ba: [B, 16] → view as [B, 4, 4] → split b/a
    ba = mixed_ba.view(B, num_heads_qk, 2 * v_group_size)  # [B, 4, 4]
    b = ba[:, :, :v_group_size].reshape(B, num_heads_v)  # [B, 8]
    a = ba[:, :, v_group_size:].reshape(B, num_heads_v)  # [B, 8]

    return mixed_qkv, z, b, a


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
):
    """Slice contiguous [q|k|v|z] / [b|a] rows into QKV, z, b and a."""
    total_q = num_heads_qk * head_qk
    total_k = num_heads_qk * head_qk
    total_v = num_heads_v * head_v
    q = mixed_qkvz[:, 0:total_q]
    k = mixed_qkvz[:, total_q : total_q + total_k]
    v = mixed_qkvz[:, total_q + total_k : total_q + total_k + total_v]
    z = mixed_qkvz[:, total_q + total_k + total_v :].reshape(
        mixed_qkvz.shape[0], num_heads_v, head_v
    )
    # b/a are sliced views of mixed_ba; copy so the public contract of returning
    # independent tensors holds and the caller's input stays untouched.
    return (
        torch.cat([q, k, v], dim=-1),
        z,
        mixed_ba[:, 0:num_heads_v].contiguous(),
        mixed_ba[:, num_heads_v:].contiguous(),
    )
