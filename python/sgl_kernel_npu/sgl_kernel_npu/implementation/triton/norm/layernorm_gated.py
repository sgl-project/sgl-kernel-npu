# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/layernorm_gated.py to support npu
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

import triton
import triton.language as tl


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

    # Load weight once (broadcasted over rows)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    for row_block_start in tl.range(pid_m * BLOCK_M, M, BLOCK_M * tl.num_programs(0)):
        # Compute row indices for this program
        rows = row_block_start + tl.arange(0, BLOCK_M)
        row_mask = rows < M

        # Load X: shape [BLOCK_M, BLOCK_N]
        x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)

        # Load Z if needed
        if HAS_Z:
            z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
            z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(
                tl.float32
            )
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
