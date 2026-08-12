from enum import Enum
from typing import Optional, Sequence

import torch


class TransferDirection(Enum):
    H2D = 1
    D2H = 2


class TransferFlag(Enum):
    FAST2D = 2


def transfer_state_dim_exchange(
    device_states: Sequence[torch.Tensor],
    host_states: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    direction: TransferDirection,
    layer_begin: int,
    layer_count: int,
    flags: TransferFlag = TransferFlag.FAST2D,
) -> None:
    """Submit indexed state-sidecar copies to the current NPU stream.

    Device components use ``[layers, device_slots, *state_shape]`` and host
    components use ``[host_slots, layers, 1, *state_shape]``.  A device slot
    payload may be a dense permutation (for example the NPU NEXTN temporal
    transpose); the Host payload is an opaque byte-exact backup of that physical
    layout.  The call only enqueues H2D/D2H work; completion is ordered by the
    caller's stream/event.

    Argument validation lives in the registered C++ operator so direct
    ``torch.ops`` callers and this convenience wrapper share one safety
    boundary.
    """
    torch.ops.npu.transfer_state_dim_exchange(
        list(device_states),
        list(host_states),
        device_indices,
        host_indices,
        direction.value,
        layer_begin,
        layer_count,
        flags.value,
    )


def transfer_state_per_layer_direct_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    flags: TransferFlag = TransferFlag.FAST2D,
) -> None:
    """Load one layer of page-first Host state into layer-first Device state.

    ``src`` is the complete page-first Host component and ``dst`` is the
    current layer view of the layer-first Device component. This matches the
    GPU per-layer direct PF->LF entry and is enqueued on the caller's current
    NPU stream.
    """
    torch.ops.npu.transfer_state_per_layer_direct_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        flags.value,
    )


def transfer_state_all_layer_direct_lf_pf(
    device_states: Sequence[torch.Tensor],
    host_states: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    flags: TransferFlag = TransferFlag.FAST2D,
) -> None:
    """Back up every layer of one LF Device component into PF Host state.

    This is the Ascend counterpart of the GPU all-layer direct LF->PF entry.
    The operation is enqueued on the caller's current NPU stream.
    """
    torch.ops.npu.transfer_state_all_layer_direct_lf_pf(
        list(device_states),
        list(host_states),
        device_indices,
        host_indices,
        flags.value,
    )


def transfer_kv_dim_exchange(
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    device_k: torch.Tensor,
    host_k: torch.Tensor,
    device_v: torch.Tensor,
    host_v: torch.Tensor,
    device_index_k: Optional[torch.Tensor] = None,
    host_index_k: Optional[torch.Tensor] = None,
    page_size: int = 128,
    direction: TransferDirection = TransferDirection.H2D,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    """
    In the L1 and L2 radix cache scenarios, perform batch copy of KV data between the device and the host.

    Args:
        device_indices: token indices in device
        host_indices: token indices in host
        device_k: k_buffer in device
        host_k: k_buffer in host
        device_v: v_buffer in device
        host_v: v_buffer in host
        device_index_k: index_k_buffer in device
        host_index_k: index_k_buffer in host
        page_size: page size
        direction: only support H2D and D2H.
        flags: only FAST2D is supported, which indicates 2D data transfer via calling aclrtMemcpy2dAsync.
    """
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
