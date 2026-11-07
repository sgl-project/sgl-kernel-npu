from typing import Sequence

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._types import TransferFlag

_torch_native_fn = None


@register_impl("data_move.transfer_kv_all_layer_direct_lf_pf", "auto")
@register_impl("data_move.transfer_kv_all_layer_direct_lf_pf", "aclrt")
def _aclrt(
    device_states: Sequence[torch.Tensor],
    host_states: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    torch.ops.npu.transfer_state_all_layer_direct_lf_pf(
        list(device_states),
        list(host_states),
        device_indices,
        host_indices,
        flags.value,
    )


@register_impl("data_move.transfer_kv_all_layer_direct_lf_pf", "torch_native")
def _native(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.data_move.state_transfer import (
            transfer_kv_all_layer_direct_lf_pf,
        )

        _torch_native_fn = transfer_kv_all_layer_direct_lf_pf
    return _torch_native_fn(**kwargs)


@register_operator("data_move.transfer_kv_all_layer_direct_lf_pf")
def transfer_kv_all_layer_direct_lf_pf(
    device_states: Sequence[torch.Tensor],
    host_states: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    flags: TransferFlag = TransferFlag.FAST2D,
    *,
    impl="auto",
) -> None:
    """Back up selected slots of every layer and component to host.

    Each device/host component pair must share its dtype. Device slot
    payloads must be physically dense with packed slots. Layers may have
    non-overlapping padding; host components must be contiguous and pinned.

    Args:
        device_states: Layer-first device components [layers, slots, ...].
        host_states: Corresponding pinned host components [slots, layers, 1, ...].
        device_indices: Non-negative device slot indices valid for every component, without page-size conversion.
        host_indices: Corresponding non-negative host slot indices valid for every component.
        flags: TransferFlag.FAST2D for 2D copies.
        impl: auto (ACL Runtime), aclrt, or torch_native.

    Returns:
        None. The destination is updated in place. Synchronize the submitting stream before reading it.
    """
    return get_impl_by_name("data_move.transfer_kv_all_layer_direct_lf_pf", impl)(
        device_states=device_states,
        host_states=host_states,
        device_indices=device_indices,
        host_indices=host_indices,
        flags=flags,
    )
