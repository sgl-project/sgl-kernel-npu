# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import triton
import triton.language as tl


@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit(do_not_specialize=["NB", "T", "MBS"])
def l2norm_fwd_kernel_opt(
    x,
    y,
    eps,
    NB,
    T,
    D,
    MBS,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    num_sub_blocks = tl.cdiv(MBS, BT)
    base_offset = i_t * MBS

    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BT

        p_x = tl.make_block_ptr(x, (T, D), (D, 1), (sub_offset, 0), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        b_var = tl.sum(b_x * b_x, axis=1)
        b_y = b_x * tl.rsqrt(b_var + eps)[:, None]
        p_y = tl.make_block_ptr(y, (T, D), (D, 1), (sub_offset, 0), (BT, BD), (1, 0))
        tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
