"""BF16 HC wrappers: validate model shapes and bound FP32 projection scratch.

Mix reuses the Triton sigmoid multiplication implementation.
Mix/combine use direct execution through 8192 rows and 4096-row chunks above it.
All conversions are per call, including during graph capture.
"""

import torch

from . import hc_core as core


def _tensors(values):
    if any(not isinstance(x, torch.Tensor) for x in values):
        raise ValueError("Expected tensors")
    device = values[0].device
    if device.type != "npu" or any(x.device != device for x in values):
        raise ValueError("Expected tensors on the same NPU")
    if any(x.dtype != torch.bfloat16 or not x.is_contiguous() for x in values):
        raise ValueError("Expected contiguous BF16 tensors")


def _dimensions(hc, hs):
    if type(hc) is not int or type(hs) is not int or (hc, hs) != (4, 2560):
        raise ValueError("Current model requires HC=4 and H=2560")


def mix(x, down, up, hc, hs):
    _dimensions(hc, hs)
    _tensors((x, down, up))
    if (
        x.ndim != 2
        or x.shape[1] != 10240
        or down.shape != (320, 10240)
        or up.shape != (10240, 320)
    ):
        raise ValueError("Expected x[R,10240], down[320,10240], up[10240,320]")
    if x.shape[0] == 0:
        return x.new_empty((0, hs))
    implementation = core.mix if x.shape[0] <= core.DIRECT_ROWS else core.mix_chunked
    return implementation(x, down, up, hc, hs)


def combine(block, residual, normed, weight, hc, hs):
    _dimensions(hc, hs)
    _tensors((block, residual, normed, weight))
    if (
        residual.ndim != 2
        or residual.shape[1] != 10240
        or normed.shape != residual.shape
        or block.shape != (residual.shape[0], hs)
        or weight.shape != (4, 10240)
    ):
        raise ValueError(
            "Expected block[R,2560], residual/normed[R,10240], weight[4,10240]"
        )
    if residual.shape[0] == 0:
        return torch.empty_like(residual)
    implementation = (
        core.combine if residual.shape[0] <= core.DIRECT_ROWS else core.combine_chunked
    )
    return implementation(block, residual, normed, weight, hc, hs)
