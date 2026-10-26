"""Public split_qkvgate_gemma_rmsnorm_rope with implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.split_qkvgate_gemma_rmsnorm_rope", "auto")
@register_impl("norm.split_qkvgate_gemma_rmsnorm_rope", "triton")
def _triton_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
):
    import triton

    from ...implementation.triton.norm.split_qkv_rmsnorm_rope import (
        split_qkvgate_gemma_rmsnorm_rope_kernel,
    )
    from ...utils.triton_utils import get_device_properties
    from ._qkv_inputs import prepare_qkv

    input, sin, cos, q_weight, k_weight, _, _ = prepare_qkv(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        None,
        None,
        rope_dim=rope_dim,
        gemma=True,
    )
    _, num_vectorcore = get_device_properties()

    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert KV_BLOCK_SIZE == head_dim
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    Q_GATE_BLOCK_SIZE = Q_BLOCK_SIZE * 2
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size * 2 + kv_hidden_size * 2
    q_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    k_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    v_output = torch.empty(
        batch_size, kv_hidden_size, device=input.device, dtype=input.dtype
    )
    gate_output = torch.empty(
        batch_size, q_hidden_size, device=input.device, dtype=input.dtype
    )
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    n_rows = (num_vectorcore + n_cols - 1) // n_cols
    split_qkvgate_gemma_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
        input,
        sin,
        cos,
        q_output,
        k_output,
        v_output,
        gate_output,
        q_weight,
        k_weight,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        Q_BLOCK_SIZE,
        Q_GATE_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        head_dim,
        rope_dim,
        rope_dim // 2,
        head_dim - rope_dim,
    )

    return q_output, k_output, v_output, gate_output


@register_impl("norm.split_qkvgate_gemma_rmsnorm_rope", "torch_native")
def _torch_native_impl(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
):
    from ._qkv_inputs import prepare_qkv

    input, sin, cos, q_weight, k_weight, _, _ = prepare_qkv(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        None,
        None,
        rope_dim=rope_dim,
        gemma=True,
    )
    from ...implementation.torch_native.norm.split_qkvgate_gemma_rmsnorm_rope import (
        split_qkvgate_gemma_rmsnorm_rope,
    )
    from ._qkv_inputs import reference_weights

    batch = input.shape[0]
    qgate = input[:, : 2 * q_hidden_size].reshape(
        batch, q_hidden_size // head_dim, 2 * head_dim
    )
    q, gate = qgate.split(head_dim, dim=-1)
    k = input[:, 2 * q_hidden_size : 2 * q_hidden_size + kv_hidden_size].reshape(
        batch, kv_hidden_size // head_dim, head_dim
    )
    v = input[:, 2 * q_hidden_size + kv_hidden_size :]
    sin = sin.reshape(batch, 1, rope_dim)
    cos = cos.reshape(batch, 1, rope_dim)
    qw, kw, _, _ = reference_weights(head_dim, eps, q_weight, k_weight, None, None)
    return split_qkvgate_gemma_rmsnorm_rope(
        q, k, v, gate, sin, cos, eps, qw, kw, input.dtype
    )


@register_operator("norm.split_qkvgate_gemma_rmsnorm_rope")
def split_qkvgate_gemma_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
    *,
    impl="auto"
):
    """Use the original Triton implementation by default, or the PyTorch reference."""
    return get_impl_by_name("norm.split_qkvgate_gemma_rmsnorm_rope", impl)(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        q_weight,
        k_weight,
    )
