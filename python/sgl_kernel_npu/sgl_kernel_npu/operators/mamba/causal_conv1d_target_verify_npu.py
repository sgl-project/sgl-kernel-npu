"""Public causal-convolution target-verify operator."""

from typing import Optional, Union

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

PAD_SLOT_ID = -1

_triton_fn = None


@register_impl("mamba.causal_conv1d_target_verify_npu", "auto")
@register_impl("mamba.causal_conv1d_target_verify_npu", "triton")
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
            causal_conv1d_target_verify_npu,
        )

        _triton_fn = causal_conv1d_target_verify_npu

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


@register_operator("mamba.causal_conv1d_target_verify_npu")
def causal_conv1d_target_verify_npu(
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

    ``conv_state`` remains read-only. ``intermediate_conv_window`` is updated
    in place. Linear-chain verification may overwrite ``x`` and return it as
    the output; tree verification allocates a separate output. ``impl``
    accepts ``auto`` or ``triton``.
    """
    impl_fn = get_impl_by_name("mamba.causal_conv1d_target_verify_npu", impl)
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
