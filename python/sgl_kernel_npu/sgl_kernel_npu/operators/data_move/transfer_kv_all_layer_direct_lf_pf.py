from typing import Sequence

import torch

from ._types import TransferFlag
from ..utils.registry import get_impl_by_name, register_impl, register_operator


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
    from ...implementation.torch_native.data_move.state_transfer import (
        transfer_kv_all_layer_direct_lf_pf,
    )

    return transfer_kv_all_layer_direct_lf_pf(**kwargs)


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

    Args:
        device_states: Layer-first device components [layers, slots, ...].
        host_states: Corresponding pinned host components [slots, layers, 1, ...].
        device_indices: Device slot indices (token indices for KV).
        host_indices: Matching host slot or token indices.
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
