# SPDX-License-Identifier: BSD-3-Clause
"""Queue-ordered multi-component H2D with completion-event buffer ownership."""

import threading

import torch

_tls = threading.local()


def queued_h2d_copy(components, host_indices, device_indices, page_size, coalesce=True):
    """Retain the source/destination views until the asynchronous copy completes.

    This is the lifetime wrapper used by the measured GLM-5.2 queued-copy
    extension, with its native call registered in sgl-kernel-npu. It returns
    byte/descriptor counts for diagnostics; the public exchange returns None.
    The caller must not mutate the source before the stream completes.
    """
    keep = getattr(_tls, "inflight", [])
    keep[:] = [entry for entry in keep if not entry[0].query()]
    # Record the completion event on the same device/stream selected by C++.
    with torch.npu.device(components[0][0].device):
        result = torch.ops.npu.transfer_kv_dim_exchange_h2d(
            [device for device, _ in components],
            [host for _, host in components],
            host_indices,
            device_indices,
            page_size,
            coalesce,
        )
        event = torch.npu.Event()
        event.record()
    keep.append((event, components, host_indices, device_indices))
    _tls.inflight = keep
    return result
