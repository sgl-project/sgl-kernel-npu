"""Public kda target verify."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.kda.kda_target_verify", "auto")
@register_impl("attention.kda.kda_target_verify", "triton")
def _triton_impl(
    *,
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
    cache_steps,
    scale=None,
    gates_are_preactivated=None
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.kda.kda_target_verify import (
            kda_target_verify,
        )

        _triton_fn = kda_target_verify
    return _triton_fn(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        scale=scale,
        gates_are_preactivated=gates_are_preactivated,
    )


@register_impl("attention.kda.kda_target_verify", "torch_native")
def _torch_native_impl(
    *,
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
    cache_steps,
    scale=None,
    gates_are_preactivated=None
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.kda.kda_target_verify import (
            kda_target_verify,
        )

        _torch_native_fn = kda_target_verify
    return _torch_native_fn(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        scale=scale,
        gates_are_preactivated=gates_are_preactivated,
    )


@register_operator("attention.kda.kda_target_verify")
def kda_target_verify(
    *,
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
    cache_steps,
    scale=None,
    gates_are_preactivated=None,
    impl="auto"
):
    """Runs fixed-width KDA speculative verification: normalize Q/K, decay the recurrent state per channel, apply the gated delta update, and project each updated state onto Q. Initial states are read-only. Per-step snapshots are written to selected scratch slots; negative source indices select zero state and negative destination indices skip snapshots. Raw or preactivated gates and grouped query/key heads are supported.

    Args:
        A_log (Tensor): Floating [heads] (or equivalent singleton-expanded shape) decay weights. Default: Required.
        dt_bias (Tensor): GDN: [heads]; KDA: [Hk,K] gate bias. Default: Required.
        q (Tensor): KDA: [1,tokens,Hq,K]; FIA: FP16/BF16 [tokens,QH,D]. Default: Required.
        k (Tensor): Floating [1,tokens,Hk,K] keys. Default: Required.
        v (Tensor): Floating [1,tokens,Hv,V] values; Hv divisible by Hq and Hk. Default: Required.
        a (Tensor): GDN: [tokens,heads]. KDA: [tokens,Hk,K] with optional leading singleton. Default: Required.
        b (Tensor): Floating [tokens,Hv], optionally [1,tokens,Hv] for KDA. Default: Required.
        initial_state_source (Tensor): Floating [pool,Hv,V,K] read-only initial states; strided V/K supported. Default: Required.
        initial_state_indices (Tensor): INT32/INT64 [batch] source slots; negative means zero state. Default: Required.
        intermediate_states_buffer (Tensor): Writable FP32/BF16 [scratch,cache_steps,Hv,V,K] snapshots, not aliasing initial state. Default: Required.
        intermediate_state_indices (Tensor): INT32/INT64 [batch] destination slots; negative skips writes. Default: Required.
        cache_steps (int): Positive steps per request; total tokens must be divisible by it. Default: Required.
        scale (float or None): QK scaling; KDA defaults to key_dim**-0.5 and requires positive values. Default: None.
        gates_are_preactivated (bool or None): True: a is log-decay and b is sigmoid output. None infers true when both have leading singleton dimensions. Default: None.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        [1,tokens,Hv,V], same dtype/device as v; writes snapshot buffer.
    """
    return get_impl_by_name("attention.kda.kda_target_verify", impl)(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        scale=scale,
        gates_are_preactivated=gates_are_preactivated,
    )
