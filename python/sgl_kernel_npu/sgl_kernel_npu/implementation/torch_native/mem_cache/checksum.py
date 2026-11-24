"""Independent PyTorch byte extraction and CPU zlib Adler-32 reference."""

import zlib

import torch


def adler32_strided_checksum(
    tensors: list[torch.Tensor],
    strides: list[int],
    indices: list[torch.Tensor],
) -> int:
    """Read selected bytes from owning tensors; raw integer addresses are unsupported."""
    if not (len(tensors) == len(strides) == len(indices)):
        raise ValueError("data_ptrs, strides and indices must have the same length")
    value = 1
    device = indices[0].device if indices else None
    for tensor, stride, idx in zip(tensors, strides, indices):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "torch_native requires Tensor buffers, not integer addresses"
            )
        if (
            idx.device.type != "npu"
            or idx.device != device
            or idx.ndim != 1
            or not idx.is_contiguous()
            or idx.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "indices must be contiguous 1D int32/int64 tensors on the same NPU"
            )
        if not tensor.is_contiguous() or tensor.device != device:
            raise ValueError(
                "Tensor buffers must be contiguous and on the index device"
            )
        if stride < 0:
            raise ValueError("strides must be nonnegative")
        if stride == 0 or idx.numel() == 0:
            continue
        raw = tensor.detach().cpu().view(torch.uint8).numpy().tobytes()
        for row in idx.cpu().tolist():
            start = row * stride
            if row < 0 or start + stride > len(raw):
                raise ValueError("selected bytes exceed the Tensor buffer")
            value = zlib.adler32(raw[start : start + stride], value)
    return value
