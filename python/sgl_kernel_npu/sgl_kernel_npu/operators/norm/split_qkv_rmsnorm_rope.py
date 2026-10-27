"""Public split_qkv_rmsnorm_rope with implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.split_qkv_rmsnorm_rope", "auto")
@register_impl("norm.split_qkv_rmsnorm_rope", "triton")
def _triton_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
):
    from ...implementation.triton.norm.split_qkv_rmsnorm_rope import (
        split_qkv_rmsnorm_rope_kernel,
    )
    from ...utils.triton_utils import get_device_properties

    _, num_vectorcore = get_device_properties()

    KV_BLOCK_SIZE = head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    k_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    v_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    n_rows = (num_vectorcore + n_cols - 1) // n_cols
    BIAS = q_bias is not None
    NORMS = eps is not None
    rope_dim = sin.shape[-1]
    split_qkv_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
        input,
        sin,
        cos,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        BIAS,
        NORMS,
        head_dim,
        rope_dim,
        rope_dim // 2,
        head_dim - rope_dim,
        DO_PARTIAL=(head_dim != rope_dim),
        DO_HALF=is_neox_style,
    )

    return q_output, k_output, v_output


@register_impl("norm.split_qkv_rmsnorm_rope", "torch_native")
def _torch_native_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
):

    assert q_hidden_size % kv_hidden_size == 0
    from ...implementation.torch_native.norm.split_qkv_rmsnorm_rope import (
        split_qkv_rmsnorm_rope,
    )
    from ._qkv_inputs import reference_weights, split_qkv_inputs

    q, k, v = split_qkv_inputs(input, q_hidden_size, kv_hidden_size, head_dim)
    sin = sin.reshape(input.shape[0], 1, sin.shape[-1])
    cos = cos.reshape(input.shape[0], 1, cos.shape[-1])
    weights = reference_weights(head_dim, eps, q_weight, k_weight, q_bias, k_bias)
    return split_qkv_rmsnorm_rope(
        q, k, v, sin, cos, eps, *weights, is_neox_style, input.dtype
    )


@register_operator("norm.split_qkv_rmsnorm_rope")
def split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
    *,
    impl="auto"
):
    """Use the original Triton implementation by default, or the PyTorch reference."""
    return get_impl_by_name("norm.split_qkv_rmsnorm_rope", impl)(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )
