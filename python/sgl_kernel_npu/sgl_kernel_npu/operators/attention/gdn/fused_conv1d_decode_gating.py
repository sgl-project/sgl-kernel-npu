"""Public fused decode causal conv1d + in-place state shift + GDN gating operator."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.fused_conv1d_decode_gating", "auto")
@register_impl("attention.gdn.fused_conv1d_decode_gating", "triton")
def _triton_impl(
    projected_states_qkvz,
    projected_states_ba,
    conv_weights_t,
    conv_state,
    cache_indices,
    A_log,
    dt_bias,
    num_v_heads,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_conv1d_decode_gating import (
            fused_conv1d_decode_gating,
        )

        _triton_fn = fused_conv1d_decode_gating
    return _triton_fn(
        projected_states_qkvz,
        projected_states_ba,
        conv_weights_t,
        conv_state,
        cache_indices,
        A_log,
        dt_bias,
        num_v_heads,
    )


@register_impl("attention.gdn.fused_conv1d_decode_gating", "torch_native")
def _torch_native_impl(
    projected_states_qkvz,
    projected_states_ba,
    conv_weights_t,
    conv_state,
    cache_indices,
    A_log,
    dt_bias,
    num_v_heads,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_conv1d_decode_gating import (
            fused_conv1d_decode_gating,
        )

        _torch_native_fn = fused_conv1d_decode_gating
    return _torch_native_fn(
        projected_states_qkvz,
        projected_states_ba,
        conv_weights_t,
        conv_state,
        cache_indices,
        A_log,
        dt_bias,
        num_v_heads,
    )


@register_operator("attention.gdn.fused_conv1d_decode_gating")
def fused_conv1d_decode_gating(
    projected_states_qkvz,
    projected_states_ba,
    conv_weights_t,
    conv_state,
    cache_indices,
    A_log,
    dt_bias,
    num_v_heads,
    *,
    impl="auto",
):
    """Fused decode depthwise causal conv1d (kernel=4, SiLU) + in-place state shift + GDN gating.

    auto selects Triton. Reads packed ``qkv`` from the leading ``QKV_DIM`` columns of
    ``projected_states_qkvz [B, QKVZ_DIM]``, updates ``conv_state [num_slots, 3, QKV_DIM]``
    in place for slots with ``cache_indices[b] >= 0`` (padding slots ``< 0`` pass through
    without modifying ``conv_state``), and returns ``(mixed_qkv, g, beta)`` where
    ``g [B, NV]`` is float32 log-decay and ``beta [B, NV]`` holds pre-sigmoid beta logits.
    """
    return get_impl_by_name("attention.gdn.fused_conv1d_decode_gating", impl)(
        projected_states_qkvz,
        projected_states_ba,
        conv_weights_t,
        conv_state,
        cache_indices,
        A_log,
        dt_bias,
        num_v_heads,
    )
