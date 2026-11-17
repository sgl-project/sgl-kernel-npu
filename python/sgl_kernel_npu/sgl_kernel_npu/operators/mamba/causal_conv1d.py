"""Public dispatch for token-major causal convolution with mutable state."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("mamba.causal_conv1d", "auto")
@register_impl("mamba.causal_conv1d", "ascendc")
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

    Args:
        x: Contiguous BF16/FP16 Tensor [T,D] or [B,L,D], token-major inputs.
        weight: Contiguous Tensor [K,D], same dtype as x; K is 2, 3 or 4.
        conv_states: Mutable contiguous Tensor [slots,S,D], same dtype as x;
            each live slot holds at least K-1 history rows.
        bias: Optional Tensor [D], same dtype as x; defaults to None.
        query_start_loc: Optional Tensor [B+1] of packed token boundaries,
            converted to int64; defaults to None. Required for 2D x in both modes,
            ignored for 3D x.
        cache_indices: Optional Tensor [B], converted to int64; defaults to
            None (sequence b uses slot b).
        has_initial_state: Optional Tensor [B], converted to int64; nonzero
            makes prefill read history. Defaults to None (zero history),
            ignored by update.
        num_accepted_tokens: Optional Tensor [B], converted to int32; selects
            speculative history for update with K=4. Defaults to None.
        activation_mode: Int, default 0; any nonzero value applies SiLU.
        pad_slot_id: Int, default -1; matching explicit cache indices skip a
            sequence, leaving its state unchanged and output undefined.
        run_mode: Int, default 0; 1 selects integrated update, others prefill.
        impl: Keyword-only str, "auto" (default), "ascendc" or "torch_native".

    Returns:
        Newly allocated Tensor with x's shape, dtype and device.

    All tensors share a device (NPU for ascendc). Empty optional tensors count
    as absent. State writes store raw input/history, preserving unused rows.
    Optional metadata/bias use dense physical spans after dtype conversion.
    See the API page for speculative history selection and writeback.
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
