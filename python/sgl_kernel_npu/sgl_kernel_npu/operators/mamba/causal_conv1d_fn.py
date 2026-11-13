"""Public causal conv1d fn."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("mamba.causal_conv1d_fn", "auto")
@register_impl("mamba.causal_conv1d_fn", "torch_native")
def _torch_native_impl(
    x,
    weight,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    conv_states=None,
    activation="silu",
    pad_slot_id=-1,
    **kwargs
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.causal_conv1d_fn import (
            causal_conv1d_fn,
        )

        _torch_native_fn = causal_conv1d_fn
    return _torch_native_fn(
        x,
        weight,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states,
        activation,
        pad_slot_id,
        **kwargs
    )


@register_operator("mamba.causal_conv1d_fn")
def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    conv_states=None,
    activation="silu",
    pad_slot_id=-1,
    *,
    impl="auto",
    **kwargs
):
    """Applies depthwise causal convolution to packed variable-length or padded batched sequences, with optional initial state, bias and SiLU. It updates selected final-state cache slots and returns the convolution result; packed input padding after the last query is zero-filled.

    Args:
        x (Tensor): Floating [B,C] or [B,C,T] for update; [C,total_tokens] or [B,C,T] for prefill. Default: Required.
        weight (Tensor): Convolution: [channels,width]. Gemma: [H], same dtype as input. Default: Required.
        bias (Tensor or None): Optional [channels] convolution bias. Default: None.
        query_start_loc (Tensor): INT32/INT64 [batch+1] cumulative packed lengths beginning at zero. Default: None.
        cache_indices (Tensor): Integer [batch] cache slot for each sequence. Default: None.
        has_initial_state (Tensor): Boolean [batch] indicating which requests use existing history. Default: None.
        conv_states (Tensor): Prefill: [pool,channels,width-1]; rollback: [layers,pool,window,channels]. Default: None.
        activation (str, bool or None): None disables activation; silu/swish enables SiLU. Update also accepts bool. Default: 'silu'.
        pad_slot_id (int or None): Padding slot sentinel; update Triton path skips matching requests. Default: -1.
        impl (str): auto selects torch_native; explicit backend selection. Default: 'auto'.
        kwargs (dict): Additional compatibility arguments, currently unused. Default: Required.

    Returns:
        Output with the same shape/dtype as x; updates conv_states.
    """
    return get_impl_by_name("mamba.causal_conv1d_fn", impl)(
        x,
        weight,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        conv_states,
        activation,
        pad_slot_id,
        **kwargs
    )
