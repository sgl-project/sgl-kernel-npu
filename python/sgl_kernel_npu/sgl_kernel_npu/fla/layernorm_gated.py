# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/layernorm_gated.py to support npu
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html


import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties
from einops import rearrange

def rms_norm(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    upcast=True,
):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    mean = None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        weight = weight.to(x.dtype)
        out, inv_rms = torch_npu.npu_rms_norm(x, weight, eps)
        if bias is not None:
            out = out + bias
        rstd_flat = inv_rms.reshape(-1)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        rstd_flat = (rstd.squeeze(-1).transpose(0, 1).contiguous().view(-1))
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype), mean, rstd_flat

# TODO:
# - Convert int32 comparison to fp32
# - Increase BLOCK size on M-axis
@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel_npu_smid(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    core = tl.program_id(0)
    group = tl.program_id(1)
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    for row in tl.range(core, M, tl.num_programs(0)):
        start_x = X + row * stride_x_row + group * N
        start_y = Y + row * stride_y_row + group * N
        if HAS_Z:
            start_z = Z + row * stride_z_row + group * N
        # Compute mean and variance
        cols = tl.arange(0, BLOCK_N)
        x = tl.load(start_x + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(start_z + cols, mask=cols < N).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean + row, mean)
            xbar = tl.where(cols < N, x - mean, 0.0)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(cols < N, x, 0.0)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)
        # Normalize and apply linear transformation
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(start_z + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        # Write output
        tl.store(start_y + cols, y, mask=mask)


def layer_norm_fwd_npu(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)

    if not is_rms_norm:
        raise NotImplementedError("LayerNorm not implemented yet")
    out_native, mean, rstd = rms_norm(
        x=x,
        weight=weight,
        bias=bias,
        z=z,
        eps=eps,
        group_size=None if group_size == N else group_size,
        norm_before_gate=norm_before_gate,
        upcast=True,
    )
    if out is not None:
        out.copy_(out_native)
    else:
        out = out_native
    return out, mean, rstd
