"""Independent PyTorch reference for the BF16 gate multiply and output cast.

This reference promotes ``y`` to FP32, multiplies by the FP32 ``gate`` and casts
the product back to ``y``'s dtype, materializing the full-size intermediates.
It shares no code with the Triton kernel, so it is usable as an accuracy
reference. The Triton backend is defined to reproduce this expression bitwise.
"""


def gate_multiply(y, gate):
    """Return ``(y.float() * gate).to(y.dtype)`` without launching any kernel.

    ``y`` is the BF16 normalized activation and ``gate`` is the FP32 sigmoid
    gate; both keep the public operator's shape, dtype and device contract.
    The result is a newly allocated tensor and the inputs are not modified.
    """
    return (y.float() * gate).to(y.dtype)
