import torch

from ._types import TransferFlag
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("data_move.transfer_kv_per_layer_direct_pf_lf", "auto")
@register_impl("data_move.transfer_kv_per_layer_direct_pf_lf", "aclrt")
def _aclrt(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    torch.ops.npu.transfer_state_per_layer_direct_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        flags.value,
    )


@register_impl("data_move.transfer_kv_per_layer_direct_pf_lf", "torch_native")
def _native(**kwargs):
    from ...implementation.torch_native.data_move.state_transfer import (
        transfer_kv_per_layer_direct_pf_lf,
    )

    return transfer_kv_per_layer_direct_pf_lf(**kwargs)


@register_operator("data_move.transfer_kv_per_layer_direct_pf_lf")
def transfer_kv_per_layer_direct_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    flags: TransferFlag = TransferFlag.FAST2D,
    *,
    impl="auto",
) -> None:
    """Restore selected slots into one device layer.

    Args:
        src: Pinned host state [slots, layers, 1, ...].
        dst: Destination NPU layer view [slots, ...].
        src_indices: Host slot indices in copy order.
        dst_indices: Corresponding device slots.
        layer_id: Zero-based source layer.
        flags: TransferFlag.FAST2D for 2D copies.
        impl: auto (ACL Runtime), aclrt, or torch_native.

    Returns:
        None. The destination is updated in place. Synchronize the submitting stream before reading it.
    """
    return get_impl_by_name("data_move.transfer_kv_per_layer_direct_pf_lf", impl)(
        src=src,
        dst=dst,
        src_indices=src_indices,
        dst_indices=dst_indices,
        layer_id=layer_id,
        flags=flags,
    )
