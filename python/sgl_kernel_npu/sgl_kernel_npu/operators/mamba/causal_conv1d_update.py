"""Public causal conv1d update."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_decode_fn = None
_torch_native_fn = None


@register_impl("mamba.causal_conv1d_update", "auto")
@register_impl("mamba.causal_conv1d_update", "triton")
def _triton_impl(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    intermediate_conv_window=None,
    pad_slot_id=-1,
    metadata=None,
    validate_data=False,
):
    global _decode_fn, _triton_fn
    if validate_data:
        assert cache_seqlens is None
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation is not None:
        assert activation in ("silu", "swish")
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, _ = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1

    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    intermediate_strides = (
        (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
        if intermediate_conv_window is not None
        else (0, 0, 0, 0)
    )
    if cache_seqlens is None and num_accepted_tokens is None:
        if _decode_fn is None:
            from ...implementation.torch_native.mamba.causal_conv1d_update import (
                causal_conv1d_update_decode,
            )

            _decode_fn = causal_conv1d_update_decode
        out = _decode_fn(x, conv_state, weight, bias, activation, conv_state_indices)
        return out.squeeze(-1) if unsqueeze else out

    if _triton_fn is None:
        from ...implementation.triton.mamba.causal_conv1d_update import (
            causal_conv1d_update,
        )

        _triton_fn = causal_conv1d_update
    out = _triton_fn(
        x,
        conv_state,
        weight,
        bias,
        activation,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window,
        pad_slot_id,
        stride_state_indices=stride_state_indices,
        intermediate_strides=intermediate_strides,
    )
    return out.squeeze(-1) if unsqueeze else out


@register_impl("mamba.causal_conv1d_update", "torch_native")
def _torch_native_impl(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    intermediate_conv_window=None,
    pad_slot_id=-1,
    metadata=None,
    validate_data=False,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.causal_conv1d_update import (
            causal_conv1d_update,
        )

        _torch_native_fn = causal_conv1d_update
    return _torch_native_fn(
        x,
        conv_state,
        weight,
        bias,
        activation,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window,
        pad_slot_id,
        metadata,
        validate_data,
    )


@register_operator("mamba.causal_conv1d_update")
def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    intermediate_conv_window=None,
    pad_slot_id=-1,
    metadata=None,
    validate_data=False,
    *,
    impl="auto",
):
    """Updates depthwise convolution history for decode or speculative multi-token execution. Decode without acceptance metadata returns a new output. The speculative path writes the output into x and can also save a history window after each token. The state layout is [pool,channels,history].

    Args:
        x (Tensor): Floating [B,C] or [B,C,T] for update; [C,total_tokens] or [B,C,T] for prefill. Default: Required.
        conv_state (Tensor): Floating [pool,channels,history] writable state, enough capacity for width+steps-2. Default: Required.
        weight (Tensor): Convolution: [channels,width]. Gemma: [H], same dtype as input. Default: Required.
        bias (Tensor or None): Optional [channels] convolution bias. Default: None.
        activation (str, bool or None): None disables activation; silu/swish enables SiLU. Update also accepts bool. Default: None.
        cache_seqlens (Tensor or None): Reserved [batch] metadata; this path does not implement circular-cache indexing. Default: None.
        conv_state_indices (Tensor or None): Integer [batch] state slot mapping. Default: None.
        num_accepted_tokens (Tensor or None): Integer [batch]; speculative history starts at accepted_count-1. Default: None.
        intermediate_conv_window (Tensor or None): Writable [pool,steps,channels,width-1] per-step history. Default: None.
        pad_slot_id (int or None): Padding slot sentinel; update Triton path skips matching requests. Default: -1.
        metadata (object or None): Reserved and unused. Default: None.
        validate_data (bool): Enable the existing update metadata assertions. Default: False.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        Output with x shape/dtype; updates conv_state and optional intermediate windows, and writes x on the speculative path.
    """
    return get_impl_by_name("mamba.causal_conv1d_update", impl)(
        x,
        conv_state,
        weight,
        bias,
        activation,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window,
        pad_slot_id,
        metadata,
        validate_data,
    )
