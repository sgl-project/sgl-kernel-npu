"""Public causal-convolution target-verify operator."""

from typing import Optional, Union

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

PAD_SLOT_ID = -1

_triton_fn = None
_torch_native_fn = None


@register_impl("mamba.causal_conv1d_target_verify", "auto")
@register_impl("mamba.causal_conv1d_target_verify", "triton")
def _triton_impl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data: bool = False,
) -> torch.Tensor:
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mamba.causal_conv1d_target_verify import (
            causal_conv1d_target_verify,
        )

        _triton_fn = causal_conv1d_target_verify

    return _triton_fn(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        pad_slot_id,
        validate_data,
    )


@register_impl("mamba.causal_conv1d_target_verify", "torch_native")
def _torch_native_impl(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data: bool = False,
) -> torch.Tensor:
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.causal_conv1d_target_verify import (
            causal_conv1d_target_verify,
        )

        _torch_native_fn = causal_conv1d_target_verify
    return _torch_native_fn(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        pad_slot_id,
        validate_data,
    )


@register_operator("mamba.causal_conv1d_target_verify")
def causal_conv1d_target_verify(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data: bool = False,
    *,
    impl="auto",
) -> torch.Tensor:
    """Run target-verify convolution and write per-token state checkpoints.

    ``auto`` selects Triton; ``torch_native`` selects independent PyTorch.

    Args:
        x: torch.Tensor, shape [B, D] or [B, D, S], dtype Floating point; output uses x.dtype, and the wrapper does not enumerate a supported-dtype set. Candidate tokens. Rank 2 is treated as S = 1; B and S must be non-empty.
        conv_state: torch.Tensor, shape [cache_lines, D, state_len >= K - 1], dtype Floating point; history uses the tensor's element dtype. Committed convolution state; read only.
        weight: torch.Tensor, shape [D, K] or [K, D], dtype Floating point; loaded into FP32 computation. Per-channel convolution weight; K must be in 2..4.
        bias: torch.Tensor or None, shape [D] or —, dtype Floating point; loaded into FP32 computation. Optional per-channel bias.
        activation: bool, str, or None, shape —, dtype —. True maps to "silu"; False maps to None. Strings may be "silu" or "swish".
        conv_state_indices: torch.Tensor, shape One-dimensional, [>= B], dtype Integer; no explicit dtype check. Selects the committed state row for each batch item. An entry equal to pad_slot_id makes that candidate inactive.
        intermediate_conv_window: torch.Tensor, shape [scratch_size, steps >= S, D, window >= K - 1], dtype Floating point; no explicit supported-dtype list. Scratch windows updated in place for active candidates.
        intermediate_state_indices: torch.Tensor, shape One-dimensional, [>= B], dtype Integer; no explicit dtype check. Selects the scratch row for each batch item.
        retrieve_next_token: torch.Tensor or None, shape [rows >= B, steps >= S] or —, dtype Integer tree mapping. Enables tree mode when provided; the three retrieve_* arguments must then all be provided.
        retrieve_next_sibling: torch.Tensor or None, shape [rows >= B, steps >= S] or —, dtype Integer tree mapping. Next-sibling mapping for tree mode.
        retrieve_parent_token: torch.Tensor or None, shape [rows >= B, steps >= S] or —, dtype Integer tree mapping. Receives the derived parent mapping in tree mode; updated in place for active candidates.
        pad_slot_id: int, shape —, dtype —. Sentinel used to identify inactive state slots.
        validate_data: bool, shape —, dtype —. Enables additional stride checks; it is not a numerical validation mode.
        impl: Backend name: auto, triton or torch_native.

    Returns:
        Output shaped like x. Linear mode overwrites and aliases x; tree
        mode preserves x and returns separate storage. Writes scratch and
        tree parent mappings; conv_state stays read-only.
    """
    impl_fn = get_impl_by_name("mamba.causal_conv1d_target_verify", impl)
    return impl_fn(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        pad_slot_id,
        validate_data,
    )
