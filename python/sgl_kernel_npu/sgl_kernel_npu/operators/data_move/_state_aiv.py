"""Shared ``ascendc_aiv`` backend for the mamba state direct transfers.

The ``data_move.transfer_kv_per_layer_direct_pf_lf`` (H2D, per layer) and
``data_move.transfer_kv_all_layer_direct_lf_pf`` (D2H, all layers) operators
route here for their ``ascendc_aiv`` implementation: one fused
``state_exchange`` AIV kernel per call. Host addresses resolve through the
registered DVA table (MemFabric or aclrt registration), and the ACLNN
adapter records NPU tensors on the caller's stream with the transfer
description passed through CANN tiling.

Requirements specific to this backend: host memory must be registered with
the DVA table (register_host_memory / register_acl_host_memory) before
transfer, destination slot indices must be unique (the round-robin AIV
cores would race on repeated destinations), and NPU graph capture is not
supported.
"""

from threading import Lock
from typing import Sequence

import torch

from ..mem_cache.memfabric.host_memory import resolve_host_dva
from ._types import TransferFlag

_submit_lock = Lock()

_MAX_COMPONENTS = 16


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


def aiv_per_layer(
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


def aiv_all_layer(
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
