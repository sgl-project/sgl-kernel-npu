"""Public dispatch for token-major causal convolution with mutable state."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_GENERATIONS = (Generation.ARCH22, Generation.ARCH35)
_torch_native_fn = None


@register_impl("mamba.causal_conv1d", "auto", supported_generations=_GENERATIONS)
@register_impl("mamba.causal_conv1d", "ascendc", supported_generations=_GENERATIONS)
def _ascendc_impl(
    x,
    weight,
    conv_states,
    bias,
    query_start_loc,
    cache_indices,
    has_initial_state,
    num_accepted_tokens,
    activation_mode,
    pad_slot_id,
    run_mode,
):
    import torch

    return torch.ops.npu.causal_conv1d(
        x,
        weight,
        conv_states,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        num_accepted_tokens,
        activation_mode,
        pad_slot_id,
        run_mode,
    )


@register_impl("mamba.causal_conv1d", "torch_native")
def _torch_native_impl(
    x,
    weight,
    conv_states,
    bias,
    query_start_loc,
    cache_indices,
    has_initial_state,
    num_accepted_tokens,
    activation_mode,
    pad_slot_id,
    run_mode,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.causal_conv1d import causal_conv1d

        _torch_native_fn = causal_conv1d
    return _torch_native_fn(
        x,
        weight,
        conv_states,
        bias=bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        num_accepted_tokens=num_accepted_tokens,
        activation_mode=activation_mode,
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
    )


@register_operator("mamba.causal_conv1d")
def causal_conv1d(
    x,
    weight,
    conv_states,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    num_accepted_tokens=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
    *,
    impl="auto",
):
    """Return y and update conv_states in place; auto selects ascendc.

    x is [T, D] or [B, L, D], weight is [K, D], and conv_states is
    [slots, S, D]. K is 2, 3 or 4. A 2D x requires query_start_loc in both
    modes. run_mode=1 selects the integrated update; every other value uses
    prefill. Nonzero activation_mode applies SiLU. Padded y rows are undefined.
    See the API documentation for history selection and speculative writeback.
    """
    return get_impl_by_name("mamba.causal_conv1d", impl)(
        x,
        weight,
        conv_states,
        bias,
        query_start_loc,
        cache_indices,
        has_initial_state,
        num_accepted_tokens,
        activation_mode,
        pad_slot_id,
        run_mode,
    )
