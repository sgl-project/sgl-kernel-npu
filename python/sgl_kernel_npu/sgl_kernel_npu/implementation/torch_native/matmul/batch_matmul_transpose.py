"""Independent PyTorch reference for batch matmul with interleaved axes."""

import torch


def batch_matmul_transpose(tensor_a, tensor_b):
    """Compute the product with FP32 accumulation and return the input dtype."""
    if not all(isinstance(tensor, torch.Tensor) for tensor in (tensor_a, tensor_b)):
        raise ValueError("tensor_a and tensor_b must be tensors")
    if tensor_b.device != tensor_a.device:
        raise ValueError("both tensors must be on the same device")
    if (
        tensor_a.dtype not in (torch.float16, torch.bfloat16)
        or tensor_b.dtype != tensor_a.dtype
    ):
        raise ValueError("both tensors must have the same float16 or bfloat16 dtype")
    if tensor_a.ndim != 3 or tensor_b.ndim != 3:
        raise ValueError("both tensors must be 3D")
    if not tensor_b.is_contiguous():
        raise ValueError("tensor_b must be contiguous")
    m, batch, k = tensor_a.shape
    if m > 1024:
        raise ValueError(f"M must be <= 1024, got {m}")
    if tensor_b.shape[:2] != (batch, k):
        raise ValueError("tensor_a/tensor_b shapes do not match")
    output_elements = m * batch * tensor_b.shape[-1]
    if any(
        not 0 < count <= 2**32 - 1
        for count in (tensor_a.numel(), tensor_b.numel(), output_elements)
    ):
        raise ValueError("tensor element counts must be in [1, UINT32_MAX]")

    output = torch.bmm(tensor_a.float().transpose(0, 1), tensor_b.float())
    return output.transpose(0, 1).to(tensor_a.dtype).contiguous()
