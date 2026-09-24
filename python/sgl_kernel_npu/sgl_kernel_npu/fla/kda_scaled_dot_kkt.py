# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the flash-linear-attention KDA implementation.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.fla.utils import exp2, prepare_chunk_indices


@triton.jit(do_not_specialize=["T"])
def _chunk_kda_scaled_dot_kkt_fwd_kernel_128(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    gk_scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute one 64-token KDA triangle with two 32-token programs.

    Each program owns one diagonal tile. The second program also computes the
    strictly causal tile to its left. Gate values are in log2 space on the
    migrated Ascend path, so the bounded factorization uses ``exp2``.
    """
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    row_start = i_t * BT + i_i * BC
    if row_start >= T:
        return

    rows = tl.arange(0, BC)
    key_offsets = tl.arange(0, BK)
    row_mask = row_start + rows < T
    key_mask = key_offsets < K
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K

    beta_rows = tl.load(
        beta + (bos + row_start + rows) * H + i_h,
        mask=row_mask,
        other=0.0,
    )
    q_rows_ptr = tl.make_block_ptr(
        q,
        (T, K),
        (H * K, 1),
        (row_start, 0),
        (BC, BK),
        (1, 0),
    )
    k_rows_ptr = tl.make_block_ptr(
        k,
        (T, K),
        (H * K, 1),
        (row_start, 0),
        (BC, BK),
        (1, 0),
    )
    g_rows_ptr = tl.make_block_ptr(
        g,
        (T, K),
        (H * K, 1),
        (row_start, 0),
        (BC, BK),
        (1, 0),
    )
    q_rows = tl.load(q_rows_ptr, boundary_check=(0, 1))
    k_rows = tl.load(k_rows_ptr, boundary_check=(0, 1))
    g_rows = tl.load(g_rows_ptr, boundary_check=(0, 1)) * gk_scale

    last_diag_token = min(row_start + BC, T) - 1
    first_gate = (
        tl.load(
            g + row_start * H * K + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        * gk_scale
    )
    last_gate = (
        tl.load(
            g + last_diag_token * H * K + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        * gk_scale
    )
    diag_ref = 0.5 * (first_gate + last_gate)
    diag_k_ptr = tl.make_block_ptr(
        k,
        (K, T),
        (1, H * K),
        (0, row_start),
        (BK, BC),
        (0, 1),
    )
    diag_g_ptr = tl.make_block_ptr(
        g,
        (K, T),
        (1, H * K),
        (0, row_start),
        (BK, BC),
        (0, 1),
    )
    diag_k = tl.load(diag_k_ptr, boundary_check=(0, 1))
    diag_g = tl.load(diag_g_ptr, boundary_check=(0, 1)) * gk_scale
    col_mask = row_start + rows < T
    diag_row_decay = exp2(g_rows - diag_ref[None, :])
    diag_col_decay = tl.where(col_mask[None, :], exp2(diag_ref[:, None] - diag_g), 0.0)
    decayed_diag_k = diag_k * diag_col_decay
    diag_a = tl.dot(k_rows * diag_row_decay, decayed_diag_k)
    diag_q = tl.dot(q_rows * diag_row_decay, decayed_diag_k)
    diag_a *= beta_rows[:, None]
    diag_q *= scale
    diag_a = tl.where(rows[:, None] > rows[None, :], diag_a, 0.0)
    diag_q = tl.where(rows[:, None] >= rows[None, :], diag_q, 0.0)
    diag_a_ptr = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (row_start, i_i * BC),
        (BC, BC),
        (1, 0),
    )
    diag_q_ptr = tl.make_block_ptr(
        Aqk + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (row_start, i_i * BC),
        (BC, BC),
        (1, 0),
    )
    tl.store(
        diag_a_ptr,
        diag_a.to(diag_a_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        diag_q_ptr,
        diag_q.to(diag_q_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )

    if i_i == 1:
        ref = (
            tl.load(
                g + row_start * H * K + key_offsets,
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            * gk_scale
        )
        left_k_ptr = tl.make_block_ptr(
            k,
            (K, T),
            (1, H * K),
            (0, i_t * BT),
            (BK, BC),
            (0, 1),
        )
        left_g_ptr = tl.make_block_ptr(
            g,
            (K, T),
            (1, H * K),
            (0, i_t * BT),
            (BK, BC),
            (0, 1),
        )
        left_k = tl.load(left_k_ptr, boundary_check=(0, 1))
        left_g = tl.load(left_g_ptr, boundary_check=(0, 1)) * gk_scale
        row_decay = exp2(g_rows - ref[None, :])
        left_decay = exp2(ref[:, None] - left_g)
        decayed_left_k = left_k * left_decay
        inter_a = tl.dot(k_rows * row_decay, decayed_left_k)
        inter_q = tl.dot(q_rows * row_decay, decayed_left_k)
        inter_a *= beta_rows[:, None]
        inter_q *= scale

        inter_a_ptr = tl.make_block_ptr(
            A + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (row_start, 0),
            (BC, BC),
            (1, 0),
        )
        inter_q_ptr = tl.make_block_ptr(
            Aqk + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (row_start, 0),
            (BC, BC),
            (1, 0),
        )
        tl.store(
            inter_a_ptr,
            inter_a.to(inter_a_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            inter_q_ptr,
            inter_q.to(inter_q_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )


def chunk_kda_scaled_dot_kkt_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    gk_scale: float = 1.0,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the fused KDA causal key products used by NPU PCP.

    This optimized contract is specialized for Kimi-K3's 128-wide key state
    and 64-token chunks. Each program computes a 32-token diagonal tile; the
    second program also computes the causal tile to its left.
    """
    if q.shape != k.shape or gk.shape != k.shape:
        raise ValueError(
            "q, k, and gk must have the same [B,T,H,K] shape, got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, gk={tuple(gk.shape)}"
        )
    B, T, H, K = k.shape
    if K != 128 or chunk_size != 64:
        raise ValueError(
            "fused NPU KDA scaled-dot requires K=128 and chunk_size=64, "
            f"got K={K}, chunk_size={chunk_size}"
        )
    if tuple(beta.shape) != (B, T, H):
        raise ValueError(f"beta must have shape {(B, T, H)}, got {tuple(beta.shape)}")

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    num_chunks = (
        triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    )
    triangular = torch.zeros(B, T, H, chunk_size, device=k.device, dtype=output_dtype)
    query_key = torch.zeros_like(triangular)
    _chunk_kda_scaled_dot_kkt_fwd_kernel_128[(num_chunks, 2, B * H)](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        A=triangular,
        Aqk=query_key,
        scale=scale,
        gk_scale=gk_scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=chunk_size,
        BC=32,
        BK=128,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=8,
        num_stages=3,
    )
    return triangular, query_key
