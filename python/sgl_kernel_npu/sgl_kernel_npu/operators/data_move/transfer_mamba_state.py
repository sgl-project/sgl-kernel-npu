import torch

from ._types import TransferDirection
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("data_move.transfer_mamba_state", "auto")
@register_impl("data_move.transfer_mamba_state", "aclrt")
def _aclrt(
    device_buf: torch.Tensor,
    host_buf: torch.Tensor,
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    direction: TransferDirection = TransferDirection.H2D,
):
    torch.ops.npu.transfer_mamba_state(
        device_buf,
        host_buf,
        device_indices,
        host_indices,
        direction.value,
    )


@register_impl("data_move.transfer_mamba_state", "torch_native")
def _native(**kwargs):
    from ...implementation.torch_native.data_move.state_transfer import (
        transfer_mamba_state,
    )

    return transfer_mamba_state(**kwargs)


@register_operator("data_move.transfer_mamba_state")
def transfer_mamba_state(
    device_buf: torch.Tensor,
    host_buf: torch.Tensor,
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    direction: TransferDirection = TransferDirection.H2D,
    *,
    impl="auto",
) -> None:
    """Copy selected Mamba slots across all layers.

    Args:
        device_buf: Device state [layers, slots, *state_shape].
        host_buf: Pinned host state [slots, layers, 1, *state_shape].
        device_indices: Device slot indices (token indices for KV).
        host_indices: Matching host slot or token indices.
        direction: TransferDirection.H2D or TransferDirection.D2H.
        impl: auto (ACL Runtime), aclrt, or torch_native.

    Returns:
        None. The destination is updated in place. Synchronize the submitting stream before reading it.
    """
    return get_impl_by_name("data_move.transfer_mamba_state", impl)(
        device_buf=device_buf,
        host_buf=host_buf,
        device_indices=device_indices,
        host_indices=host_indices,
        direction=direction,
    )
