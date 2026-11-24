"""Tensor conversion and scale decoding for PyTorch references."""

import torch


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type == "npu":
        import torch_npu

        # Internal-format weights must be unpacked before a device-to-CPU copy.
        if torch_npu.get_npu_format(tensor) != 2:
            tensor = torch_npu.npu_format_cast(tensor, 2)
    return tensor.cpu()


def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 bytes, including its minimum exponent and NaN encoding."""
    bits = _cpu(scale).contiguous().view(torch.uint8).to(torch.int32)
    values = torch.exp2((bits - 127).float())
    return torch.where(bits == 255, float("nan"), values)
