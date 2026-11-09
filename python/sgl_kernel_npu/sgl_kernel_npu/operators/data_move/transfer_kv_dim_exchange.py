from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._types import TransferDirection, TransferFlag

_torch_native_fn = None


@register_impl("data_move.transfer_kv_dim_exchange", "auto")
@register_impl("data_move.transfer_kv_dim_exchange", "aclrt")
def _aclrt(
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    device_k: torch.Tensor,
    host_k: torch.Tensor,
    device_v: torch.Tensor,
    host_v: torch.Tensor,
    device_index_k: Optional[torch.Tensor] = None,
    host_index_k: Optional[torch.Tensor] = None,
    device_index_k_scale: Optional[torch.Tensor] = None,
    host_index_k_scale: Optional[torch.Tensor] = None,
    page_size: int = 128,
    direction: TransferDirection = TransferDirection.H2D,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    torch.ops.npu.transfer_kv_dim_exchange(
        device_k,
        host_k,
        device_v,
        host_v,
        device_indices,
        host_indices,
        page_size,
        direction.value,
        flags.value,
    )
    if device_index_k is not None and host_index_k is not None:
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k,
            host_index_k,
            torch.empty(0),
            torch.empty(0),
            device_indices,
            host_indices,
            page_size,
            direction.value,
            flags.value,
        )
    if device_index_k_scale is not None and host_index_k_scale is not None:
        # Device scale cache is 4-D (layers, pages, page_size, 1) while the host
        # cache is 5-D (pages, layers, page_size, 1, 1); the kernel requires both
        # operands to be 5-D, so pad the device operand with a trailing singleton.
        if device_index_k_scale.dim() == 4:
            device_index_k_scale = device_index_k_scale.unsqueeze(-1)
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k_scale,
            host_index_k_scale,
            torch.empty(0),
            torch.empty(0),
            device_indices,
            host_indices,
            page_size,
            direction.value,
            flags.value,
        )


@register_impl("data_move.transfer_kv_dim_exchange", "torch_native")
def _native(**kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.data_move.state_transfer import (
            transfer_kv_dim_exchange,
        )

        _torch_native_fn = transfer_kv_dim_exchange
    return _torch_native_fn(**kwargs)


@register_operator("data_move.transfer_kv_dim_exchange")
def transfer_kv_dim_exchange(
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    device_k: torch.Tensor,
    host_k: torch.Tensor,
    device_v: torch.Tensor,
    host_v: torch.Tensor,
    device_index_k: Optional[torch.Tensor] = None,
    host_index_k: Optional[torch.Tensor] = None,
    device_index_k_scale: Optional[torch.Tensor] = None,
    host_index_k_scale: Optional[torch.Tensor] = None,
    page_size: int = 128,
    direction: TransferDirection = TransferDirection.H2D,
    flags: TransferFlag = TransferFlag.FAST2D,
    *,
    impl="auto",
) -> None:
    """Copy selected KV pages across layer-first and page-first layouts.

    Each device/host pair must share its dtype; V must also share K dtype.
    Device page payloads may be densely transposed with packed outer pitches;
    host buffers must be contiguous and pinned. Only page-group first indices
    are used for addressing.

    Args:
        device_indices: Device slot indices (token indices for KV).
        host_indices: Matching host slot or token indices.
        device_k: Device K [layers, pages, page_size, heads, dim].
        host_k: Pinned host K [pages, layers, page_size, heads, dim].
        device_v: Device V, or an empty tensor.
        host_v: Host V, or an empty tensor.
        device_index_k: Optional device indexer key cache.
        host_index_k: Matching optional host indexer key cache.
        device_index_k_scale: Optional per-token device indexer scale (4D or 5D).
        host_index_k_scale: Matching optional 5D host scale.
        page_size: Tokens per KV page.
        direction: TransferDirection.H2D or TransferDirection.D2H.
        flags: TransferFlag.FAST2D for 2D copies.
        impl: auto (ACL Runtime), aclrt, or torch_native.

    Returns:
        None. The destination is updated in place. Synchronize the submitting stream before reading it.
    """
    return get_impl_by_name("data_move.transfer_kv_dim_exchange", impl)(
        device_indices=device_indices,
        host_indices=host_indices,
        device_k=device_k,
        host_k=host_k,
        device_v=device_v,
        host_v=host_v,
        device_index_k=device_index_k,
        host_index_k=host_index_k,
        device_index_k_scale=device_index_k_scale,
        host_index_k_scale=host_index_k_scale,
        page_size=page_size,
        direction=direction,
        flags=flags,
    )
