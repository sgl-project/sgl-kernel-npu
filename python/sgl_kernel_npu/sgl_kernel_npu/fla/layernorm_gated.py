# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/layernorm_gated.py to support npu
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html


import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


# TODO:
# - Convert int32 comparison to fp32
# - Increase BLOCK size on M-axis
@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel_npu_simd(
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


def layer_norm_fwd_npu_simd(
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
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    _, num_vectorcore = get_device_properties()
    grid = (triton.cdiv(num_vectorcore, ngroups), ngroups)
    _layer_norm_fwd_1pass_kernel_npu_simd[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        multibuffer=True,
    )
    return out, mean, rstd


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel_npu(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    # Compute row indices for this program
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # Mask for valid rows and cols
    row_mask = rows < M
    col_mask = cols < N

    # Load weight once (broadcasted over rows)
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    # Load X: shape [BLOCK_M, BLOCK_N]
    x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)

    # Load Z if needed
    if HAS_Z:
        z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
        z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

    # Compute statistics per row
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # [BLOCK_M]
        xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        tl.store(Mean + rows, mean, mask=row_mask)
    else:
        xbar = tl.where(col_mask[None, :], x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N

    rstd = 1.0 / tl.sqrt(var + eps)  # [BLOCK_M]
    tl.store(Rstd + rows, rstd, mask=row_mask)

    # Normalize
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :]
    if HAS_BIAS:
        y += b[None, :]

    # Post-gate
    if HAS_Z and NORM_BEFORE_GATE:
        y *= z * tl.sigmoid(z)

    # Store output
    y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
    tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


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
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("Feature dim too large.")

    # Choose BLOCK_M: e.g., 16, 32, 64 — depends on NPU vector core capacity
    BLOCK_M = 64  # Tune this based on your NPU's register/shared memory

    # Now grid is (num blocks over M, num groups)
    grid = (triton.cdiv(M, BLOCK_M), ngroups)
    _layer_norm_fwd_1pass_kernel_npu[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        # Remove multibuffer if not needed
    )
    return out, mean, rstd
