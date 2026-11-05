"""Public split_qkv_tp_rmsnorm_rope operator with explicit implementation selection."""

import torch
import torch.distributed

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


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
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.split_qkv_tp_rmsnorm_rope import (
            split_qkv_tp_rmsnorm_rope,
        )

        _triton_fn = split_qkv_tp_rmsnorm_rope
    return _triton_fn(
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
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.split_qkv_tp_rmsnorm_rope import (
            split_qkv_tp_rmsnorm_rope,
        )

        _torch_native_fn = split_qkv_tp_rmsnorm_rope
    return _torch_native_fn(
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
    """Split a local QKV projection, apply tensor-parallel RMSNorm to Q and K, then rotate their leading channels using NeoX RoPE.

    Args:
        input: float32 / float16 / bfloat16 tensor of shape [B, Q + 2*K]. Dense local
            projection in Q, K, V order. Q and K denote the local widths.
        cos: floating point tensor of shape [B, R]. Dense cosine rows. The first R/2 entries of
            each row apply to every head.
        sin: floating point tensor of shape [B, R]. Dense sine rows, paired with cos.
        q_hidden_size: Local Q width (Q), a multiple of the local K width.
        kv_hidden_size: Local width (K) of each of K and V. Local widths must contain whole
            heads.
        head_dim: Channels per head, a power of two.
        eps: Added to the mean square before normalization.
        q_weight: floating point tensor of shape [Q]. Local Q channel weights.
        k_weight: floating point tensor of shape [K]. Local K channel weights.
        rotary_dim: Number of leading channels rotated per head (R), even and at most head_dim.
        tp_world: Number of ranks in the TP group. Use 1 on a single device.
        tp_group: Use None on one device. With multiple ranks, all group members must call
            collectively with matching batch sizes and shard widths.
        impl: Keyword-only. Choices are "auto", "triton", "torch_native". Defaults to
            "triton" via "auto".

    Returns:
        New tensors with input dtype and device, shaped [B, Q], [B, K] and [B, K].
    """
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
