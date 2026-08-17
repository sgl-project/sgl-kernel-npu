from enum import Enum
from typing import Optional

import torch


class TransferDirection(Enum):
    H2D = 1
    D2H = 2


class TransferFlag(Enum):
    FAST2D = 2


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


def transfer_mamba_state(
    device_buf: torch.Tensor,
    host_buf: torch.Tensor,
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    direction: TransferDirection = TransferDirection.H2D,
):
    """
    Transfer Mamba/SSM state between device (layer-first) and host (page-first).

    Device buffer layout: [num_layers, device_size, *state_shape]  (layer-first)
    Host buffer layout:   [host_size, num_layers, 1, *state_shape] (page-first)

    Uses aclrtMemcpy2dAsync for efficient 2D strided copy, transferring all
    layers for each slot index in a single 2D copy call.

    Args:
        device_buf: device Mamba state buffer [num_layers, size, *shape]
        host_buf: host Mamba state buffer [size, num_layers, 1, *shape]
        device_indices: slot indices in device buffer
        host_indices: slot indices in host buffer
        direction: H2D (host→device) or D2H (device→host)
    """
    torch.ops.npu.transfer_mamba_state(
        device_buf,
        host_buf,
        device_indices,
        host_indices,
        direction.value,
    )
