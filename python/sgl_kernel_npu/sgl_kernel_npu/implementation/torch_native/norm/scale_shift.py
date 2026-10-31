"""Independent PyTorch implementation of the legacy scale/shift semantics."""

import torch


def _promoted_dtype(left, right):
    # Triton prefers float16 when the two 16-bit float formats are mixed.
    # PyTorch otherwise supplies the same promotion for our floating inputs.
    if {left.dtype, right.dtype} == {torch.float16, torch.bfloat16}:
        return torch.float16
    return torch.promote_types(left.dtype, right.dtype)


def fused_scale_shift(x, scale, shift, scale_constant=1.0):
    """Compute scale/shift without importing or calling Triton.

    Inputs are assumed valid. Flatten modulation tensors just as the Triton
    wrapper does; element count, rather than original shape, selects the path.
    """
    scale = scale.reshape(-1)
    shift = shift.reshape(-1)
    if shift.numel() == x.numel():
        constant = scale_constant
        shift = shift.reshape(x.shape).float()
    else:
        constant = 1.0
        if shift.numel() != 1:
            shift = shift.float()

    # Keep the addition in scale.dtype before applying tensor promotion.
    factor = constant + scale
    product_dtype = _promoted_dtype(x, factor)
    product = x.to(product_dtype) * factor.to(product_dtype)
    output_dtype = _promoted_dtype(product, shift)
    output = product.to(output_dtype) + shift.to(output_dtype)
    return output.to(x.dtype)
