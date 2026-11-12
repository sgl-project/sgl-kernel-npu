"""Grouped Gemma RMSNorm using the existing normalization kernel."""

import math

import torch
import triton

from ..utils.triton_utils import get_device_properties
from .layer_norm_fwd import _layer_norm_fwd_1pass_kernel_npu


def grouped_gemma_rmsnorm(x, weight, group_size, eps):
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise ValueError("Expected tensors")
    if x.device.type != "npu" or weight.device != x.device:
        raise ValueError("Expected tensors on the same NPU")
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (x, weight)):
        raise ValueError("Expected contiguous BF16 tensors")
    if x.ndim != 2 or x.shape[1] != 10240 or weight.shape != (10240,):
        raise ValueError("Expected x[R,10240], weight[10240]")
    if type(group_size) is not int or group_size != 2560:
        raise ValueError("Expected group_size=2560")
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected finite positive epsilon")

    rows, hidden = x.shape
    out = torch.empty_like(x)
    if rows == 0:
        return out
    groups = hidden // group_size
    # Convert inside each invocation so graph replay sees weight updates.
    effective_weight = weight.float() + 1.0
    rstd = torch.empty(groups * rows, dtype=torch.float32, device=x.device)

    # Width 2560 pads to 4096. Four rows exceed arch22 UB; two rows fit.
    block_m = 2
    block_n = triton.next_power_of_2(group_size)
    _, num_vectorcore = get_device_properties()
    grid = (min(triton.cdiv(rows, block_m), num_vectorcore), groups)
    _layer_norm_fwd_1pass_kernel_npu[grid](
        x,
        out,
        effective_weight,
        None,  # no bias
        None,  # no gate
        None,  # RMSNorm does not produce means
        rstd,
        x.stride(0),
        out.stride(0),
        0,
        rows,
        group_size,
        eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NORM_BEFORE_GATE=True,
        IS_RMS_NORM=True,
    )
    return out
