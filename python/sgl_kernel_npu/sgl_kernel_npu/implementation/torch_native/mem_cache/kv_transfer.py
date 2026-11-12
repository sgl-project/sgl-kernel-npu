"""Synchronous independent Torch reference for in-place page transfers.

This reference directly indexes original buffers. It does not reuse production
layer views, pointer metadata, MemFabric copy APIs or the Triton implementation.
"""

import torch


def transfer_kv(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layout,
    direction,
    kv_layers,
    index_layers
):
    if page_size <= 0 or device_indices.ndim != 1 or host_indices.ndim != 1:
        raise ValueError("expected positive page_size and one-dimensional indices")
    if device_indices.dtype != torch.int64 or host_indices.dtype != torch.int64:
        raise ValueError("KV indices must have dtype int64")
    if (
        device_indices.numel() != host_indices.numel()
        or device_indices.numel() % page_size
    ):
        raise ValueError("indices must describe equal numbers of complete pages")
    if host_layout not in ("page_first", "layer_first") or direction not in (1, 2):
        raise ValueError("invalid Host layout or transfer direction")
    if device_k is None or host_k is None or not device_k.numel() or not host_k.numel():
        raise ValueError("K buffers must be nonempty")
    if device_v is None or host_v is None:
        raise ValueError("V buffers must be tensors")
    # Host destinations must not be read while earlier asynchronous work runs.
    torch.npu.current_stream(device_k.device).synchronize()
    di, hi = device_indices.cpu().tolist(), host_indices.cpu().tolist()
    pairs = [(device_k, host_k, kv_layers)]
    if device_v.numel():
        pairs.append((device_v, host_v, kv_layers))
    if index_layers[2] != 0:
        pairs.extend(
            (dev, host, index_layers)
            for dev, host in (
                (device_index_k, host_index_k),
                (device_index_k_scale, host_index_k_scale),
            )
            if dev is not None and host is not None and dev.numel() and host.numel()
        )
    for dev, host, (dstart, hstart, count) in pairs:
        count = dev.shape[0] - dstart if count is None else count
        host_layers = host.shape[1] if host_layout == "page_first" else host.shape[0]
        if (
            dstart < 0
            or hstart < 0
            or count < 0
            or dstart + count > dev.shape[0]
            or hstart + count > host_layers
        ):
            raise ValueError("KV layer selection exceeds Device or Host storage")
        for offset in range(count):
            dl, hl = dstart + offset, hstart + offset
            for pos in range(0, len(di), page_size):
                dt, ht = di[pos], hi[pos]
                if dt < 0 or ht < 0 or dt % page_size or ht % page_size:
                    raise ValueError("page starts must be nonnegative and aligned")
                dp, hp = dt // page_size, ht // page_size
                device_page = dev[dl, dp]
                host_page = (
                    host[hp, hl] if host_layout == "page_first" else host[hl, hp]
                )
                if device_page.ndim == 2:  # Four-dimensional Device scale storage.
                    host_page = host_page.squeeze(-1)
                if direction == 1:
                    device_page.copy_(host_page)
                else:
                    host_page.copy_(device_page.cpu())
    torch.npu.current_stream(device_k.device).synchronize()
