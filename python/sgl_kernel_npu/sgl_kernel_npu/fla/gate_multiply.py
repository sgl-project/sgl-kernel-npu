"""Preserve BF16 normalization rounding while fusing FP32 multiply and cast."""

from functools import cache

import torch
import triton
import triton.language as tl


@cache
def _vector_cores(device_index):
    return triton.runtime.driver.active.utils.get_device_properties(device_index)[
        "num_vectorcore"
    ]


@triton.jit(do_not_specialize=["N"])
def _multiply(Y, G, O, N, B: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(N, B), tl.num_programs(0)):
        offsets = block * B + tl.arange(0, B)
        y = tl.load(Y + offsets, offsets < N, other=0).to(tl.float32)
        g = tl.load(G + offsets, offsets < N, other=0)
        tl.store(O + offsets, y * g, offsets < N)


def gate_multiply(y, gate):
    """Compute ``(y.float() * gate).to(torch.bfloat16)`` on an NPU.

    y must be BF16, gate FP32, and shapes/devices identical. Compute the
    gate with the existing sigmoid implementation before calling this
    helper. The input y retains the original BF16 normalization rounding;
    this helper does not fuse normalization or approximate sigmoid.

    Contiguous inputs with >= 16,777,216 elements use the fused kernel;
    smaller or strided inputs retain the PyTorch expression. This threshold
    was measured on 910C; it is not a claim of benefit on every NPU.
    Inference only: no backward implementation is provided.
    """
    if y.device.type != "npu" or y.device != gate.device:
        raise ValueError("y and gate must be on the same NPU")
    if (
        y.dtype != torch.bfloat16
        or gate.dtype != torch.float32
        or y.shape != gate.shape
    ):
        raise ValueError("expected matching BF16 y and FP32 gate shapes")
    if y.requires_grad or gate.requires_grad:
        raise ValueError("gate_multiply is inference only")
    if y.numel() < 16777216 or not y.is_contiguous() or not gate.is_contiguous():
        return (y.float() * gate).to(y.dtype)
    with torch.npu.device(y.device):
        out = torch.empty_like(y)
        cores = _vector_cores(y.device.index)
        _multiply[(cores,)](y, gate, out, y.numel(), B=4096, enable_fp_fusion=False)
        return out
