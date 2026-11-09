"""Independent PyTorch batch matmul, including explicit NZ unpacking."""

import torch


def batch_matmul_transpose(
    tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None
):
    """Fill the caller's output buffer, preserving the legacy None return value."""
    format_mode = "ND" if format_mode is None else format_mode
    quant_mode = "per_channel_symm" if quant_mode is None else quant_mode
    if format_mode not in ("ND", "NZ"):
        raise ValueError(f"Unsupported format_mode: {format_mode}")
    if quant_mode not in ("per_channel_symm", "per_channel_asymm", "per_token_symm"):
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")
    tensors = (tensor_a, tensor_b, tensor_c)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("tensor_a, tensor_b and tensor_c must be tensors")
    if any(tensor.device != tensor_a.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if tensor_a.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != tensor_a.dtype for tensor in tensors
    ):
        raise ValueError("all tensors must have the same float16 or bfloat16 dtype")
    nz = format_mode == "NZ"
    if tensor_a.ndim != 3 or tensor_c.ndim != 3 or tensor_b.ndim != (4 if nz else 3):
        raise ValueError("invalid tensor rank for the selected format_mode")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")
    if any(not 0 < tensor.numel() <= 2**32 - 1 for tensor in tensors):
        raise ValueError("tensor element counts must be in [1, UINT32_MAX]")
    m, batch, k = tensor_a.shape
    if m > 1024:
        raise ValueError(f"M must be <= 1024, got {m}")
    if tensor_b.shape[0] != batch or tensor_b.shape[2 if nz else 1] != k:
        raise ValueError("tensor_a/tensor_b shapes do not match")
    if nz and (tensor_b.shape[3] != 16 or k % 16):
        raise ValueError("NZ requires [B, N/16, K, 16] with K divisible by 16")
    n = tensor_b.shape[1] * 16 if nz else tensor_b.shape[2]
    if tensor_c.shape != (m, batch, n):
        raise ValueError("tensor_c must have shape [M, B, N]")
    # Contiguous spans allow exact overlap checks, including offset views.
    begin = tensor_c.data_ptr()
    end = begin + tensor_c.numel() * tensor_c.element_size()
    for tensor in (tensor_a, tensor_b):
        other_begin = tensor.data_ptr()
        other_end = other_begin + tensor.numel() * tensor.element_size()
        if begin < other_end and other_begin < end:
            raise ValueError("tensor_c must not overlap an input")
    weights = tensor_b.permute(0, 2, 1, 3).reshape(batch, k, n) if nz else tensor_b
    # The custom kernel accumulates FP16/BF16 operands in FP32.
    output = torch.bmm(tensor_a.float().transpose(0, 1), weights.float())
    tensor_c.copy_(output.transpose(0, 1).to(tensor_c.dtype))
