"""Unified normalization operators with independent, lazily loaded backends."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("norm.fused_split_qk_norm", "auto")
@register_impl("norm.fused_split_qk_norm", "triton")
def _triton_impl(
    fused_qkv_a_proj_out,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_dim,
    eps=1e-06,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.fused_split_qk_norm import (
            fused_split_qk_norm,
        )

        _triton_fn = fused_split_qk_norm
    return _triton_fn(
        fused_qkv_a_proj_out,
        q_a_layernorm,
        kv_a_layernorm,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_dim,
        eps,
    )


_torch_native_fn = None


@register_impl("norm.fused_split_qk_norm", "torch_native")
def _torch_native_impl(
    fused_qkv_a_proj_out,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_dim,
    eps=1e-06,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.fused_split_qk_norm import (
            fused_split_qk_norm,
        )

        _torch_native_fn = fused_split_qk_norm
    return _torch_native_fn(
        fused_qkv_a_proj_out,
        q_a_layernorm,
        kv_a_layernorm,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_dim,
        eps,
    )


@register_operator("norm.fused_split_qk_norm")
def fused_split_qk_norm(
    fused_qkv_a_proj_out,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_dim,
    eps=1e-06,
    *,
    impl="auto"
):
    """Split [N, Rq+Rkv+Drope] and RMS-normalize the Q/K latent parts.

    auto selects Triton. q_a_layernorm and kv_a_layernorm expose floating
    weight tensors of lengths Rq/Rkv and optional independent bias tensors.
    eps is a positive scalar and overrides any epsilon on those objects.
    Returns new tensors [N, Rq], [N, 1, Rkv], [N, 1, Drope] with input dtype.

    impl is keyword-only: "auto", "triton", or "torch_native".
    """
    impl_fn = get_impl_by_name("norm.fused_split_qk_norm", impl)
    return impl_fn(
        fused_qkv_a_proj_out,
        q_a_layernorm,
        kv_a_layernorm,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_dim,
        eps,
    )
