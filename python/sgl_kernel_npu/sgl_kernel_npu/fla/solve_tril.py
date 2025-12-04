# Adapted and Merge from
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/solve_tril.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, By Triton_Ascend & sglang_ascend

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel_npu.fla.utils import (
    exp,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    safe_exp,
)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_paral(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16
    base_t = i_t * LARGE_BLOCK_T
    tl.device_print("i_t:", i_t)
    tl.device_print("base_t:", base_t)

    b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)  # (N_BLOCKS, 16, 16)
    for blkid in range(0, N_BLOCKS):
        row_start_o = base_t + blkid * 16
        col_start_o = row_start_o % BT
        p_A_subrec16 = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (row_start_o, col_start_o), (16, 16), (1, 0)
        )
        b_A_subrec16 = tl.load(p_A_subrec16, boundary_check=(0, 1)).to(
            tl.float32
        )  # (16, 16)
        b_A = tl.insert_slice(
            ful=b_A,
            sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
            offsets=[blkid, 0, 0],
            sizes=[1, 16, 16],
            strides=[1, 1, 1],
        )

    # load multi 16x16 into UB
    local_ori_A = tl.trans(b_A, (1, 0, 2))
    local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))  # (16, N_BLOCKS*16)

    tmp = tl.arange(0, 16).to(tl.float32)
    rows = tmp[:, None]
    cols = tmp[None, :]
    is_lower = (rows > cols).to(b_A.dtype)
    b_A = -b_A * is_lower

    o_i = tl.arange(0, 16)
    for i in range(1, 16):

        nblks_vec16 = -tl.extract_slice(
            local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
        )
        b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

        dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
        dot_product = tl.sum(dot_tmp, 0)
        b_a = b_a + dot_product  # (N_BLOCKS, 16)

        row_mask = o_i == i  # (16,), True at position i
        update_mask = row_mask[None, :, None]  # (1, 16, 1)
        b_a_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
        b_A = tl.where(update_mask, b_a_expanded, b_A)  # shape keeps (N_BLOCKS, 16, 16)

    on_diagonal = rows == cols
    b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

    b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
    p_Ai = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0)
    )
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_paral_v3(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    base_t = i_t * LARGE_BLOCK_T

    NTASKS: tl.constexpr = 2
    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS

    for taskid in range(0, NTASKS):
        base_t += taskid * (LARGE_BLOCK_T // NTASKS)

        b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)  # (N_BLOCKS, 16, 16)
        for blkid in range(0, N_BLOCKS):
            row_start_o = base_t + blkid * 16
            col_start_o = row_start_o % BT
            # using ptr with mask instead of tl.load(block_ptr)
            offs_rows_in_block = tl.arange(0, 16)
            offs_cols_in_block = tl.arange(0, 16)
            # strides (H*BT, 1)
            ptr_A_subrec16 = (
                A
                + row_start_o * H * BT
                + col_start_o
                + offs_rows_in_block[:, None] * H * BT
                + offs_cols_in_block[None, :]
            )
            global_rows = row_start_o + offs_rows_in_block[:, None]
            global_cols = col_start_o + offs_cols_in_block[None, :]
            load_mask = (global_rows < T) & (global_cols < BT)
            b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask, other=0.0).to(
                tl.float32
            )
            b_A = tl.insert_slice(
                ful=b_A,
                sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
                offsets=[blkid, 0, 0],
                sizes=[1, 16, 16],
                strides=[1, 1, 1],
            )

        # load multi 16x16
        local_ori_A = tl.trans(b_A, (1, 0, 2))
        local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))  # (16, N_BLOCKS*16)

        # change mask into matrix elementwise action
        tmp = tl.arange(0, 16).to(tl.float32)
        rows = tmp[:, None]
        cols = tmp[None, :]
        is_lower = (rows > cols).to(b_A.dtype)
        b_A = -b_A * is_lower

        for i in range(1, 16):

            nblks_vec16 = -tl.extract_slice(
                local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
            )
            b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

            dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
            dot_product = tl.sum(dot_tmp, 0)
            b_a = b_a + dot_product  # (N_BLOCKS, 16)

            b_a_new_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
            b_A = tl.insert_slice(
                ful=b_A,
                sub=b_a_new_expanded,
                offsets=[0, i, 0],
                sizes=[N_BLOCKS, 1, 16],
                strides=[1, 1, 1],
            )

        on_diagonal = rows == cols
        b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

        b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
        p_Ai = tl.make_block_ptr(
            Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0)
        )
        # using ptr with mask instead of tl.load(block_ptr)
        offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
        offs_cols_to_store = tl.arange(0, 16)
        # strides (H*16, 1)
        p_Ai = (
            Ad
            + base_t * H * 16
            + 0
            + offs_rows_to_store[:, None] * H * 16
            + offs_cols_to_store[None, :]
        )
        global_store_rows = base_t + offs_rows_to_store[:, None]
        store_mask = global_store_rows < T
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=store_mask,
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 32
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    p_A_21 = tl.make_block_ptr(
        A, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel_reorder_all_masked(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t_val = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        i_t = i_t_val
    else:
        bos, eos = i_b * T, i_b * T + T

    # Base pointers (already offset by batch and head)
    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    # ------------------ Load Ai_22 (Ad block at row i_t*64+16, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_22 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # ------------------ Load A_21 (A block at row i_t*64+16, col 0, 16x16) ------------------
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)  # A has 64 cols
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_22, A_21, input_precision="ieee")

    # ------------------ Load Ai_11 (Ad block at row i_t*64, col 0, 16x16) ------------------
    offs_m = i_t * 64 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_11 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_21 = -tl.dot(tmp, Ai_11, input_precision="ieee")

    # ------------------ Load Ai_44 (Ad block at row i_t*64+48, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 48 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_44 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # ------------------ Load A_43 (A block at row i_t*64+48, col 32, 16x16) ------------------
    offs_n = 32 + tl.arange(0, 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_43 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_44, A_43, input_precision="ieee")

    # ------------------ Load Ai_33 (Ad block at row i_t*64+32, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_33 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_43 = -tl.dot(tmp, Ai_33, input_precision="ieee")

    # ------------------ Build Ai_22_32 (32x32) ------------------
    Ai_22_32 = tl.zeros((32, 32), tl.float32)
    Ai_22_32 = tl.insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
    Ai_22_32 = tl.insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
    Ai_22_32 = tl.insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

    # ------------------ Load A_21_32 (A block at row i_t*64+32, col 0, 32x32) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21_32 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")

    # ------------------ Build Ai_11_32 (32x32) ------------------
    Ai_11_32 = tl.zeros((32, 32), tl.float32)
    Ai_11_32 = tl.insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
    Ai_11_32 = tl.insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
    Ai_11_32 = tl.insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

    Ai_21_32 = -tl.dot(tmp, Ai_11_32, input_precision="ieee")

    # ------------------ Store Ai_11_32 to (i_t*64, 0) ------------------
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Store Ai_22_32 to (i_t*64+32, 32) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Store Ai_21_32 to (i_t*64+32, 0) ------------------
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Zero out the upper-right 32x32 block (rows 0~31, cols 32~63) ------------------
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < BT)  # BT=64
    ptr_Ai = Ai + offs_m[:, None] * (H * BT) + offs_n[None, :]
    zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element_ty)
    tl.store(ptr_Ai, zero_block, mask=mask_store)


def solve_tril_npu(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the lower triangular matrix
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, K]
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor.
            Default: None.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape
    Ad = torch.empty(
        B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype
    )

    LARGE_BLOCK_T = 608 * 2
    # assert A.shape[1]%LARGE_BLOCK_T == 0 # or last N_BLOCKS have not enough block which leads to tl.arange failed

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T)
        if cu_seqlens is not None
        else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, LARGE_BLOCK_T)
    solve_tril_16x16_kernel_paral_v3[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        LARGE_BLOCK_T=LARGE_BLOCK_T,
        num_warps=1,
        num_stages=4,
    )

    if BT == 16:
        return Ad

    Ai = torch.zeros_like(A, device=A.device, dtype=output_dtype)
    merge_fn = (
        merge_16x16_to_32x32_inverse_kernel
        if BT == 32
        else merge_16x16_to_64x64_inverse_kernel_reorder_all_masked
    )
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    merge_fn[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        num_warps=4,
        num_stages=3,
    )
    return Ai
