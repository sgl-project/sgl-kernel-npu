"""Mamba state direct transfers on the caller's current NPU stream.

The signatures mirror the CUDA-aligned direct calls used by the Mamba host
pool: per-layer H2D (host page-first -> device layer view) and all-layer D2H
(device layer-first -> host page-first, one submission for main plus draft
components with independent layer counts).

The ``ascendc_aiv`` implementation submits one fused ``state_exchange`` AIV
kernel per call: host addresses resolve through the registered DVA table
(MemFabric or aclrt registration), and the ACLNN adapter records NPU
tensors on the caller's stream with the transfer description passed through
CANN tiling. NPU graph capture is not supported by this AIV path. Host
memory must be registered with the DVA table (register_host_memory /
register_acl_host_memory) before transfer.
"""

from threading import Lock
from typing import Sequence

import torch

from ..mem_cache.memfabric.host_memory import resolve_host_dva
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._types import TransferFlag

_submit_lock = Lock()

_MAX_COMPONENTS = 16
_NATIVE_PER_LAYER_FN = None
_NATIVE_ALL_LAYER_FN = None


def _require_fast2d(flags):
    if flags.value & 2 != 2:
        raise ValueError("state direct transfer currently requires FAST2D (flags=2)")


def _check_indices(device_indices, host_indices):
    for index in (device_indices, host_indices):
        if index.ndim != 1 or index.dtype != torch.int64:
            raise ValueError("transfer indices must be 1-D int64 tensors")
    if device_indices.numel() != host_indices.numel():
        raise ValueError("device and host indices must have the same length")


def _check_slot_values(
    device_indices, host_indices, device_slots, host_slots, host_is_dst
):
    # The AIV kernel dereferences device-resident slot indices unchecked, so
    # pull the values once (a small sync) and mirror the torch_native
    # reference: an out-of-range slot would corrupt unrelated GM/DVA memory.
    # Duplicate sources are fine; duplicate destinations race across the
    # round-robin cores, so they are rejected.
    pairs = list(
        zip(
            device_indices.cpu().long().flatten().tolist(),
            host_indices.cpu().long().flatten().tolist(),
        )
    )
    for d, h in pairs:
        if not (0 <= d < device_slots and 0 <= h < host_slots):
            raise RuntimeError("index exceeds component slot count")
    destinations = [h for _, h in pairs] if host_is_dst else [d for d, _ in pairs]
    if len(set(destinations)) != len(destinations):
        raise RuntimeError("destination slot indices must be unique")


# ---------------------------------------------------------------------------
# H2D per layer: host [slots, layers, 1, *payload] -> device layer view
# [slots, *payload]
# ---------------------------------------------------------------------------


def _submit_aiv_per_layer(src, dst, src_indices, dst_indices, layer_id):
    # Normalized state_exchange component pair: device [1, slots, *payload],
    # host layer-narrowed [slots, 1, 1, *payload]; base pointers and strides
    # carry the offsets.

    device_view = dst.unsqueeze(0)
    host_view = src.narrow(1, layer_id, 1)
    with _submit_lock, torch.npu.device(dst_indices.device):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("AIV state transfer does not support NPU graph capture")
        torch.ops.npu.state_exchange(
            dst_indices.contiguous(),
            src_indices.contiguous(),
            [device_view],
            [host_view],
            [resolve_host_dva(host_view)],
            1,
        )


@register_impl("data_move.transfer_state_per_layer_direct_pf_lf", "auto")
@register_impl("data_move.transfer_state_per_layer_direct_pf_lf", "ascendc_aiv")
def _aiv_per_layer(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    _require_fast2d(flags)
    _check_indices(dst_indices, src_indices)
    if not dst_indices.numel():
        return
    if src.device.type != "cpu" or dst.device.type != "npu":
        raise ValueError("state source must be host memory and destination must be NPU")
    if not 0 <= layer_id < src.size(1):
        raise ValueError("layer_id is outside the host state layer count")
    _check_slot_values(
        dst_indices, src_indices, dst.size(0), src.size(0), host_is_dst=False
    )
    _submit_aiv_per_layer(src, dst, src_indices, dst_indices, layer_id)


@register_impl("data_move.transfer_state_per_layer_direct_pf_lf", "torch_native")
def _native_per_layer(**kwargs):
    global _NATIVE_PER_LAYER_FN
    if _NATIVE_PER_LAYER_FN is None:
        from ...implementation.torch_native.data_move.state_transfer import (
            transfer_state_per_layer_direct_pf_lf,
        )

        _NATIVE_PER_LAYER_FN = transfer_state_per_layer_direct_pf_lf
    return _NATIVE_PER_LAYER_FN(**kwargs)


@register_operator("data_move.transfer_state_per_layer_direct_pf_lf")
def transfer_state_per_layer_direct_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    flags: TransferFlag = TransferFlag.FAST2D,
    *,
    impl="auto",
) -> None:
    """Restore selected slots of one host layer into a device layer view.

    Args:
        src: Host state [slots, layers, 1, ...].
        dst: Destination NPU layer view [slots, ...].
        src_indices: Host slot indices in copy order.
        dst_indices: Corresponding device slots.
        layer_id: Zero-based source layer.
        flags: TransferFlag.FAST2D for 2D copies.
        impl: auto, ascendc_aiv, or torch_native.

    Returns:
        None. The destination is updated in place. Synchronize the submitting stream before reading it.
    """
    return get_impl_by_name("data_move.transfer_state_per_layer_direct_pf_lf", impl)(
        src=src,
        dst=dst,
        src_indices=src_indices,
        dst_indices=dst_indices,
        layer_id=layer_id,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# D2H all layers: device components [layers, slots, ...] -> host components
# [slots, layers, 1, ...]; components may carry independent layer counts
# (e.g. main [L, slots, ...] plus draft [1, slots, ...] pairs).
# ---------------------------------------------------------------------------


def _check_all_layer_components(device_states, host_states):
    if not device_states or len(device_states) != len(host_states):
        raise ValueError("expected matching nonempty component lists")
    if len(device_states) > _MAX_COMPONENTS:
        raise ValueError(
            f"at most {_MAX_COMPONENTS} components are supported, got {len(device_states)}"
        )
    for which, (dev, host) in enumerate(zip(device_states, host_states)):
        if dev.device.type != "npu":
            raise ValueError(f"component {which}: device state must be an NPU tensor")
        if host.device.type != "cpu":
            raise ValueError(f"component {which}: host state must be a CPU tensor")


@register_impl("data_move.transfer_state_all_layer_direct_lf_pf", "auto")
@register_impl("data_move.transfer_state_all_layer_direct_lf_pf", "ascendc_aiv")
def _aiv_all_layer(
    device_states: Sequence[torch.Tensor],
    host_states: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    _require_fast2d(flags)
    _check_indices(device_indices, host_indices)
    _check_all_layer_components(device_states, host_states)
    if not device_indices.numel():
        return
    device_states = list(device_states)
    host_states = list(host_states)
    _check_slot_values(
        device_indices,
        host_indices,
        min(dev.size(1) for dev in device_states),
        min(host.size(0) for host in host_states),
        host_is_dst=True,
    )
    with _submit_lock, torch.npu.device(device_indices.device):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("AIV state transfer does not support NPU graph capture")
        torch.ops.npu.state_exchange(
            device_indices.contiguous(),
            host_indices.contiguous(),
            device_states,
            host_states,
            [resolve_host_dva(host) for host in host_states],
            2,
        )


@register_impl("data_move.transfer_state_all_layer_direct_lf_pf", "torch_native")
def _native_all_layer(**kwargs):
    global _NATIVE_ALL_LAYER_FN
    if _NATIVE_ALL_LAYER_FN is None:
        from ...implementation.torch_native.data_move.state_transfer import (
            transfer_state_all_layer_direct_lf_pf,
        )

        _NATIVE_ALL_LAYER_FN = transfer_state_all_layer_direct_lf_pf
    return _NATIVE_ALL_LAYER_FN(**kwargs)


@register_operator("data_move.transfer_state_all_layer_direct_lf_pf")
def transfer_state_all_layer_direct_lf_pf(
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
    payloads must be physically dense with packed slots. Components may
    carry independent layer counts (main plus draft pools); host components
    may be layer-narrowed views of one registered pool.

    Args:
        device_states: Layer-first device components [layers, slots, ...].
        host_states: Corresponding host components [slots, layers, 1, ...].
        device_indices: Non-negative device slot indices valid for every component.
        host_indices: Corresponding non-negative host slot indices valid for every component.
        flags: TransferFlag.FAST2D for 2D copies.
        impl: auto, ascendc_aiv, or torch_native.

    Returns:
        None. The destination is updated in place. Synchronize the submitting stream before reading it.
    """
    return get_impl_by_name("data_move.transfer_state_all_layer_direct_lf_pf", impl)(
        device_states=device_states,
        host_states=host_states,
        device_indices=device_indices,
        host_indices=host_indices,
        flags=flags,
    )
