import torch


def clear_mamba_slots(buf: torch.Tensor, indices: torch.Tensor) -> None:
    """Reference PyTorch implementation that zeroes dim-1 slots across all layers."""
    if buf.ndim < 2:
        raise ValueError(f"buf must have at least 2 dimensions, got {buf.ndim}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got {indices.ndim}D")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indices must be int32 or int64, got {indices.dtype}")
    if buf.device != indices.device:
        raise ValueError(
            f"buf and indices must be on the same device, got {buf.device} and {indices.device}"
        )
    if indices.numel() == 0 or buf.numel() == 0:
        return
    buf.index_fill_(1, indices.to(torch.int64), 0)
