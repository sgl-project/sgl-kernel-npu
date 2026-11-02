"""Independent PyTorch implementation of fused scale/shift."""

import torch


def _promoted_dtype(left, right):
    # Triton prefers float16 when the two 16-bit float formats are mixed.
    # PyTorch otherwise supplies the same promotion for our floating inputs.
    if {left.dtype, right.dtype} == {torch.float16, torch.bfloat16}:
        return torch.float16
    return torch.promote_types(left.dtype, right.dtype)


def fused_scale_shift(x, scale, shift, scale_constant=1.0):
    """Compute scale/shift without importing or calling Triton.

    Inputs are assumed valid. Modulation tensors are flattened, and element
    counts select broadcasting and the full-input-shift branch.
    """
    scale = scale.reshape(-1)
    shift = shift.reshape(-1)
    full_shift = shift.numel() == x.numel()
    if full_shift:
        constant = scale_constant
        shift = shift.reshape(x.shape).float()
    else:
        constant = 1.0
        if shift.numel() != 1:
            shift = shift.float()

    # The broadcast kernel's single-value load keeps scalar BF16 rounding.
    # Vector scale loads retain FP32 factor precision. The full-shift kernel
    # always uses a vector load, including when the hidden size is one.
    vector_scale = full_shift or scale.numel() != 1
    factor_scale = (
        scale.float() if scale.dtype == torch.bfloat16 and vector_scale else scale
    )
    factor = constant + factor_scale
    product_dtype = _promoted_dtype(x, factor)
    product = x.to(product_dtype) * factor.to(product_dtype)
    output_dtype = _promoted_dtype(product, shift)
    output = product.to(output_dtype) + shift.to(output_dtype)
    return output.to(x.dtype)
