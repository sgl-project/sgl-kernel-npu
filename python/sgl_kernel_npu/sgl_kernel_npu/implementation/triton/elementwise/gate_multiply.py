"""Fused BF16 gate multiply and output cast for GDN (Triton NPU backend).

This backend preserves the BF16 normalization rounding of ``y`` while fusing
the FP32 promotion, the FP32 multiply by ``gate`` and the final BF16 store into
a single Triton NPU kernel. It removes the full-size promoted-``y`` and FP32
product intermediates that the plain PyTorch expression materializes.
"""

import torch
import triton
import triton.language as tl

from ..utils.triton_utils import get_device_properties

# Contiguous inputs with at least this many elements use the fused kernel;
# smaller or strided inputs fall back to the PyTorch expression. The threshold
# was measured on 910C and is not a claim of benefit on every NPU.
FUSION_MIN_ELEMENTS = 16777216


@triton.jit(do_not_specialize=["N"])
def _multiply(Y, G, O, N, B: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(N, B), tl.num_programs(0)):
        offsets = block * B + tl.arange(0, B)
        y = tl.load(Y + offsets, offsets < N, other=0).to(tl.float32)
        g = tl.load(G + offsets, offsets < N, other=0)
        tl.store(O + offsets, y * g, offsets < N)


def gate_multiply(y, gate):
    """Compute ``(y.float() * gate).to(torch.bfloat16)`` on an NPU.

    ``y`` must be BF16, ``gate`` FP32, with identical shapes and the same NPU
    device. The caller computes ``gate`` with its existing sigmoid before this
    helper; the BF16 normalization rounding already applied to ``y`` is kept.
    Normalization is not fused and sigmoid is not approximated.

    Contiguous inputs with at least ``FUSION_MIN_ELEMENTS`` elements use the
    fused kernel; smaller or strided inputs retain the PyTorch expression, which
    is bitwise identical. Inference only: no backward implementation is provided.
    """
    if (
        y.numel() < FUSION_MIN_ELEMENTS
        or not y.is_contiguous()
        or not gate.is_contiguous()
    ):
        return (y.float() * gate).to(y.dtype)
    with torch.npu.device(y.device):
        out = torch.empty_like(y)
        _, num_vectorcore = get_device_properties()
        _multiply[(num_vectorcore,)](
            y, gate, out, y.numel(), B=4096, enable_fp_fusion=False
        )
        return out
