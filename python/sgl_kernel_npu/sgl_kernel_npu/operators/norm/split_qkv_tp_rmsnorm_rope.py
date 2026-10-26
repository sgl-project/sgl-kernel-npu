"""Public split_qkv_tp_rmsnorm_rope operator with parameter preparation in registered adapters."""

from typing import Optional

import torch
import torch.distributed as dist

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.split_qkv_tp_rmsnorm_rope", "auto")
@register_impl("norm.split_qkv_tp_rmsnorm_rope", "triton")
def _triton_impl(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused kernel for splitting QKV, applying Tensor Parallel RMSNorm, and RoPE.

    This kernel performs the following operations in a fused manner:
    1. Split the concatenated QKV input into separate Q, K, V tensors
    2. Compute local variance for Q and K (for distributed RMSNorm)
    3. All-reduce the variance across TP ranks
    4. Apply global RMSNorm to Q and K using the synchronized variance
    5. Apply Neox-style Rotary Position Embedding (RoPE) to Q and K

    The input tensor layout is: [Q_hidden_size, K_hidden_size, V_hidden_size]
    where K and V share the same hidden size (kv_hidden_size).

    Args:
        input: Input tensor of shape (batch_size, q_hidden_size + 2 * kv_hidden_size)
            containing concatenated Q, K, V projections.
        cos: Cosine values for RoPE, shape (batch_size, rotary_dim).
        sin: Sine values for RoPE, shape (batch_size, rotary_dim).
        q_hidden_size: Hidden size for Q projection.
        kv_hidden_size: Hidden size for K and V projections.
        head_dim: Dimension of each attention head. Must be a power of 2.
        eps: Epsilon value for RMSNorm numerical stability.
        q_weight: Weight tensor for Q RMSNorm, shape (q_hidden_size,).
        k_weight: Weight tensor for K RMSNorm, shape (kv_hidden_size,).
        rotary_dim: Dimension for rotary position embedding. Usually equals head_dim.
        tp_world: Total number of tensor parallel ranks.
        tp_group: Process group for tensor parallel all-reduce communication.

    Returns:
        A tuple of (q, k, v) tensors:
            - q: Query tensor of shape (batch_size, q_hidden_size)
            - k: Key tensor of shape (batch_size, kv_hidden_size)
            - v: Value tensor of shape (batch_size, kv_hidden_size)

    Note:
        - This kernel is designed for Tensor Parallelism where each rank processes
          a portion of the hidden dimension.
        - The variance computation uses Kahan summation for numerical stability.
        - V is not normalized or rotated, only split from the input.
    """
    import triton

    from ...implementation.triton.norm.split_qkv_tp_rmsnorm_rope import (
        _apply_global_rmsnorm_kernel,
        _split_qkv_and_compute_local_qk_var_kernel,
    )
    from ...utils.triton_utils import get_device_properties
    from ._tp_norm_inputs import prepare_inputs

    input, cos, sin, q_weight, k_weight = prepare_inputs(
        input,
        cos,
        sin,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        q_weight,
        k_weight,
        rotary_dim,
        tp_world,
        tp_group,
    )
    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert KV_BLOCK_SIZE == head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    q = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    if batch_size == 0:
        return q, k, v
    _, num_vectorcore = get_device_properties()
    grid = (min(batch_size, num_vectorcore),)
    q_cols = q_hidden_size
    k_cols = kv_hidden_size
    q_num_heads = q_hidden_size // head_dim
    k_num_heads = kv_hidden_size // head_dim

    qk_var = torch.empty(batch_size, 2, dtype=torch.float32, device=q.device)
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        input,
        q,
        k,
        v,
        qk_var,
        batch_size,
        q_cols,
        k_cols,
        triton.next_power_of_2(q_cols),
        triton.next_power_of_2(k_cols),
        KV_BLOCK_SIZE,
        Q_BLOCK_SIZE,
    )
    if tp_world > 1:
        dist.all_reduce(qk_var, group=tp_group)

    _apply_global_rmsnorm_kernel[grid](
        q,
        k,
        cos,
        sin,
        rotary_dim,
        q_weight,
        k_weight,
        qk_var,
        eps,
        1.0 / tp_world,
        batch_size,
        q_cols,
        k_cols,
        q_num_heads,
        k_num_heads,
        head_dim,
        triton.next_power_of_2(q_cols),
        triton.next_power_of_2(k_cols),
        triton.next_power_of_2(q_num_heads),
        triton.next_power_of_2(k_num_heads),
        triton.next_power_of_2(rotary_dim // 2),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )

    return q, k, v


@register_impl("norm.split_qkv_tp_rmsnorm_rope", "torch_native")
def _torch_native_impl(
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
    from ...implementation.torch_native.norm.split_qkv_tp_rmsnorm_rope import (
        apply_global,
        compute_local,
    )
    from ._tp_norm_inputs import prepare_inputs

    input, cos, sin, q_weight, k_weight = prepare_inputs(
        input,
        cos,
        sin,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        q_weight,
        k_weight,
        rotary_dim,
        tp_world,
        tp_group,
    )
    batch = input.shape[0]
    q = torch.empty((batch, q_hidden_size), device=input.device, dtype=input.dtype)
    k = torch.empty((batch, kv_hidden_size), device=input.device, dtype=input.dtype)
    v = torch.empty_like(k)
    if batch == 0:
        return q, k, v
    qk_var = torch.empty((batch, 2), device=input.device, dtype=torch.float32)
    q_block = q_hidden_size // kv_hidden_size * head_dim
    compute_local(
        input[:, :q_hidden_size],
        input[:, q_hidden_size : q_hidden_size + kv_hidden_size],
        input[:, q_hidden_size + kv_hidden_size :],
        q,
        k,
        v,
        qk_var,
        q_block,
        head_dim,
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


@register_operator("norm.split_qkv_tp_rmsnorm_rope")
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
    *,
    impl="auto"
):
    """Fused kernel for splitting QKV, applying Tensor Parallel RMSNorm, and RoPE.

    This kernel performs the following operations in a fused manner:
    1. Split the concatenated QKV input into separate Q, K, V tensors
    2. Compute local variance for Q and K (for distributed RMSNorm)
    3. All-reduce the variance across TP ranks
    4. Apply global RMSNorm to Q and K using the synchronized variance
    5. Apply Neox-style Rotary Position Embedding (RoPE) to Q and K

    The input tensor layout is: [Q_hidden_size, K_hidden_size, V_hidden_size]
    where K and V share the same hidden size (kv_hidden_size).

    Args:
        input: Input tensor of shape (batch_size, q_hidden_size + 2 * kv_hidden_size)
            containing concatenated Q, K, V projections.
        cos: Cosine values for RoPE, shape (batch_size, rotary_dim).
        sin: Sine values for RoPE, shape (batch_size, rotary_dim).
        q_hidden_size: Hidden size for Q projection.
        kv_hidden_size: Hidden size for K and V projections.
        head_dim: Dimension of each attention head. Must be a power of 2.
        eps: Epsilon value for RMSNorm numerical stability.
        q_weight: Weight tensor for Q RMSNorm, shape (q_hidden_size,).
        k_weight: Weight tensor for K RMSNorm, shape (kv_hidden_size,).
        rotary_dim: Dimension for rotary position embedding. Usually equals head_dim.
        tp_world: Total number of tensor parallel ranks.
        tp_group: Process group for tensor parallel all-reduce communication.

    Returns:
        A tuple of (q, k, v) tensors:
            - q: Query tensor of shape (batch_size, q_hidden_size)
            - k: Key tensor of shape (batch_size, kv_hidden_size)
            - v: Value tensor of shape (batch_size, kv_hidden_size)

    Note:
        - This kernel is designed for Tensor Parallelism where each rank processes
          a portion of the hidden dimension.
        - The variance computation uses Kahan summation for numerical stability.
        - V is not normalized or rotated, only split from the input.

    impl selects auto (Triton), triton, or torch_native."""
    return get_impl_by_name("norm.split_qkv_tp_rmsnorm_rope", impl)(
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
    )
