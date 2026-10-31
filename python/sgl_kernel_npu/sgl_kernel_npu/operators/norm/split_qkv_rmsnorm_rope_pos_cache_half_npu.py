"""Unified normalization operators with independent, lazily loaded backends."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", "auto")
@register_impl("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", "triton")
def _triton_impl(
    input_tensor,
    positions,
    cos_sin_cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    rope_dim=None,
    cast_norm_to_bf16=True,
    wide_grid_min_tokens=1024,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
            split_qkv_rmsnorm_rope_pos_cache_half_npu,
        )

        _triton_fn = split_qkv_rmsnorm_rope_pos_cache_half_npu
    return _triton_fn(
        input_tensor,
        positions,
        cos_sin_cache,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        rope_dim,
        cast_norm_to_bf16,
        wide_grid_min_tokens,
    )


_torch_native_fn = None


@register_impl("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", "torch_native")
def _torch_native_impl(
    input_tensor,
    positions,
    cos_sin_cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    rope_dim=None,
    cast_norm_to_bf16=True,
    wide_grid_min_tokens=1024,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
            split_qkv_rmsnorm_rope_pos_cache_half_npu,
        )

        _torch_native_fn = split_qkv_rmsnorm_rope_pos_cache_half_npu
    return _torch_native_fn(
        input_tensor,
        positions,
        cos_sin_cache,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        rope_dim,
        cast_norm_to_bf16,
        wide_grid_min_tokens,
    )


@register_operator("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu")
def split_qkv_rmsnorm_rope_pos_cache_half_npu(
    input_tensor,
    positions,
    cos_sin_cache,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    rope_dim=None,
    cast_norm_to_bf16=True,
    wide_grid_min_tokens=1024,
    *,
    impl="auto"
):
    """Split [N, Q+2K], optionally normalize Q/K, and rotate using a packed cache.

    auto selects Triton. positions holds N int32/int64 indices; the cache is
    [max_seq, R], packed [cos_half, sin_half]. R defaults to head_dim.
    eps=None skips normalization; otherwise q_weight/k_weight need head_dim
    floats. Optional q_bias/k_bias are paired. cast_norm_to_bf16 controls the
    pre-RoPE BF16 rounding; Q/K always undergo a final BF16 rounding.
    wide_grid_min_tokens selects Triton's launch grid only. Returns new Q/K/V
    tensors [N, Q], [N, K], [N, K] in input dtype, without modifying inputs.

    impl is keyword-only: "auto", "triton", or "torch_native".
    """
    impl_fn = get_impl_by_name("norm.split_qkv_rmsnorm_rope_pos_cache_half_npu", impl)
    return impl_fn(
        input_tensor,
        positions,
        cos_sin_cache,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        rope_dim,
        cast_norm_to_bf16,
        wide_grid_min_tokens,
    )
