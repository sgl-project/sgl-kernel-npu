# Adapted and Merge from 
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/chunk_delta_h.py
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/chunk_o.py
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/chunk_scaled_dot_kkt.py
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/cumsum.py
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/solve_tril.py
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, By Triton_Ascend & sglang_ascend

from typing import Optional, Tuple, List, Union
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sgl_kernel_npu.fla.utils import prepare_chunk_indices, prepare_chunk_offsets
from sgl_kernel_npu.fla.utils import exp, safe_exp



"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++ chunk_delata_h part ++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu_kernel(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K

    b_h1_bv1 = tl.zeros([128, 64], dtype=tl.float32)
    b_h1_bv2 = tl.zeros([128, 64], dtype=tl.float32)

    v_start1 = 0
    v_start2 = 64

    offs_k = tl.arange(0, 128)[:, None]
    offs_v1 = v_start1 + tl.arange(0, 64)[None, :]
    offs_v2 = v_start2 + tl.arange(0, 64)[None, :]
    mask_kv1 = (offs_k < K) & (offs_v1 < V)
    mask_kv2 = (offs_k < K) & (offs_v2 < V)


    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        ptr_h0_bv1 = h0_ptr + offs_k * V + offs_v1 * 1
        b_h1_bv1 += tl.load(ptr_h0_bv1, mask=mask_kv1, other=0.0).to(tl.float32)

        ptr_h0_bv2 = h0_ptr + offs_k * V + offs_v2 * 1
        b_h1_bv2 += tl.load(ptr_h0_bv2, mask=mask_kv2, other=0.0).to(tl.float32)

    for i_t in range(NT):
        h_base = h + (boh + i_t) * H * K * V + i_h * K * V

        p_h1_bv1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_h1_bv1, b_h1_bv1.to(p_h1_bv1.dtype.element_ty), boundary_check=(0, 1))

        p_h1_bv2 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_h1_bv2, b_h1_bv2.to(p_h1_bv2.dtype.element_ty), boundary_check=(0, 1))

        offs_t_wv = (i_t * BT + tl.arange(0, BT))[:, None]
        offs_k_wv = tl.arange(0, 128)[None, :]
        mask_w = (offs_t_wv < T) & (offs_k_wv < K)

        w_base = w + bos * H * K + i_h * K
        ptr_w = w_base + offs_t_wv * stride_w + offs_k_wv * 1
        b_w = tl.load(ptr_w, mask=mask_w, other=0.0)

        # =============load k==================

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        v_new_base = v_new + bos * H * V + i_h * V

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        offs_t = i_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        g_ptr = g + bos + i_h * T_max 
        b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = exp(b_g_last)

        offs_t_v = (i_t * BT + tl.arange(0, BT))[:, None]
        mask_v1 = (offs_t_v < T) & (offs_v1 < V)

        v_base = v + bos * H * V + i_h * V
        ptr_v1 = v_base + offs_t_v * stride_v + offs_v1 * 1
        b_v1 = tl.load(ptr_v1, mask=mask_v1, other=0.0)
        b_v_new1 = b_v1.to(tl.float32)
        b_v_new1 -= tl.dot(b_w, b_h1_bv1.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new1 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start1), (BT, 64), (1, 0))
            tl.store(p_v_new1, b_v_new1.to(p_v_new1.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new1 = b_v_new1 * b_g[:, None]
            b_h1_bv1 = b_h1_bv1 * b_g_last

        b_v_new1 = b_v_new1.to(k.dtype.element_ty)
        b_h1_bv1 += tl.dot(b_k, b_v_new1)

        mask_v2 = (offs_t_v < T) & (offs_v2 < V)
        ptr_v2 = v_base + offs_t_v * stride_v + offs_v2 * 1
        b_v2 = tl.load(ptr_v2, mask=mask_v2, other=0.0)
        b_v_new2 = b_v2.to(tl.float32)
        b_v_new2 -= tl.dot(b_w, b_h1_bv2.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new2 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start2), (BT, 64), (1, 0))
            tl.store(p_v_new2, b_v_new2.to(p_v_new2.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new2 = b_v_new2 * b_g[:, None]
            b_h1_bv2 = b_h1_bv2 * b_g_last

        b_v_new2 = b_v_new2.to(k.dtype.element_ty)
        b_h1_bv2 += tl.dot(b_k, b_v_new2)

    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht1_bv1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_ht1_bv1, b_h1_bv1.to(p_ht1_bv1.dtype.element_ty), boundary_check=(0, 1))

        p_ht1_bv2 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_ht1_bv2, b_h1_bv2.to(p_ht1_bv2.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None
    # Pre-transpose tensors outside the kernel to ensure contiguous memory access within the kernel
    # avoiding scattered (non-contiguous) access that may lead to axis expansion
    g = g.transpose(1, 2).contiguous()
    def grid(meta):
        return (1, N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_npu_kernel[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new, final_state


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++ chunk_o part ++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o_npu_kernel(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = T

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    
    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V

    for i_t in range(NT):
        i_tg = boh + i_t
        h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(
                q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")

            p_k = tl.make_block_ptr(
                k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
            )
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

            p_h = tl.make_block_ptr(
                h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
            )
            # [BK, BV]
            b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
            
            # [BT, BK] @ [BK, BV] -> [BT, BV]
            b_o += tl.dot(b_q, b_h)
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)
        
        if USE_G:
            offs_t = i_t * BT + tl.arange(0, BT)
            mask_t = offs_t < T
            g_ptr = g + bos + i_h * T_max 
            b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)
            
            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

        o_i = tl.arange(0, BT).to(tl.float32)
        m_A = o_i[:, None] >= o_i[None, :]
        b_A = tl.where(m_A, b_A, 0)

        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_o = tl.make_block_ptr(
            o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        # to fix mma -> mma layout conversion
        # already solved by triton v3.2 or higher
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    # BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = (
            len(cu_seqlens) - 1,
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    # Pre-transpose tensors outside the kernel to ensure contiguous memory access within the kernel
    # avoiding scattered (non-contiguous) access that may lead to axis expansion
    g = g.transpose(1, 2).contiguous()
    chunk_fwd_kernel_o_npu_kernel[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_offsets,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=128,
        num_warps=4,
        num_stages=2,
    )
    return o


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++ chunk_scaled_dot_kkt part +++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_trans_kernel(
        k,
        beta,  # [H, B, T]
        g_cumsum,  # [H, B, T]
        A,
        cu_seqlens,
        chunk_indices,
        T,
        B,
        H: tl.constexpr,
        Hg: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        USE_G: tl.constexpr,
):
    bt_stride = B * T  # get from raw T
    i_t_i, _ = tl.program_id(0), tl.program_id(1)
    # i_b, i_h = i_bh // H, i_bh % H
    for i_bh in range(B * H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_i * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_i * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T
            i_t = i_t_i
        o_t = tl.arange(0, BT)  # vector<0-BT>
        o_t_fp32 = o_t.to(tl.float32)

        p_beta = tl.make_block_ptr(
            beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )  # (H, B*T) -> (BT)
        b_beta = tl.load(p_beta, boundary_check=(0,))  # vector<0-BT>

        b_A = tl.zeros([BT, BT], dtype=tl.float32)  # mat<0-BT, 0-BT>(1)
        # partial K(128) into BK(64)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * Hg + i_h // (H // Hg)) * K,
                (T, K),
                (Hg * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )  # (B*T, Hg, K) -> (BT, BK)(stride:Hg * K, 1)
            b_k = tl.load(p_k, boundary_check=(0, 1))  # (BT, BK) dot (BK, BT) part
            b_A = b_A + tl.dot(b_k, tl.trans(b_k))  # (BT, BT)

        if USE_G:
            p_g = tl.make_block_ptr(
                g_cumsum + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,)
            )  # (H, B*T) -> (BT)
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_diff = b_g[:, None] - b_g[None, :]
            b_A = b_A * safe_exp(b_g_diff)

        b_A = b_A * b_beta[:, None]
        b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)
        p_A = tl.make_block_ptr(
            A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd_npu(
        k: torch.Tensor,
        beta: torch.Tensor,
        g_cumsum: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g_cumsum (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`.
            Default: None
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """

    B, T, Hg, K = k.shape  # (1, 6, 8, 128)

    H = beta.shape[-1]
    BT = chunk_size
    if cu_seqlens is not None:
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
        )
    else:
        chunk_indices = None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

    chunk_scaled_dot_kkt_fwd_trans_kernel[(NT, 1)](
        k=k,
        beta=torch.permute(beta, (2, 0, 1)).contiguous(),
        g_cumsum=torch.permute(g_cumsum, (2, 0, 1)).contiguous(),
        # beta=beta,
        # g_cumsum=g_cumsum,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BK=128,
        num_warps=8,
        num_stages=3,
        multibuffer=True,
    )
    return A


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++ cumsum part ++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,         # 每个BLOCK的T维度size, 是CHUNK_SIZE的倍数
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr=64,   # 每个chunk的大小, 增加的新参数,与算法相关
): 
    # g_block: 当前prograd 的全局block idx
    i_block, i_b = tl.program_id(0), tl.program_id(1)
    N_CHUNKS: tl.constexpr = BLOCK_T // CHUNK_SIZE
    
    if IS_VARLEN:
        # [当前block在当前序列的idx, 当前序列的bos对应的idx]
        i_s, i_block = tl.load(chunk_indices + i_block * 2).to(tl.int32), tl.load(
            chunk_indices + i_block * 2 + 1
        ).to(tl.int32)
        
        bos, eos = tl.load(cu_seqlens + i_s).to(tl.int32), tl.load(
            cu_seqlens + i_s + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    
    if HEAD_FIRST:
        ptr_s = tl.make_block_ptr(s + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0))
        ptr_o = tl.make_block_ptr(o + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0))
        b_s = tl.load(ptr_s,  boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (H, N_CHUNKS, CHUNK_SIZE))
        b_s = tl.trans(b_s, (2, 0, 1))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (2, 0, 1))
        b_o = tl.reshape(b_o, (H, BLOCK_T))
    else:
        ptr_s = tl.make_block_ptr(s + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        ptr_o = tl.make_block_ptr(o + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        b_s = tl.load(ptr_s,  boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
        b_s = tl.trans(b_s, (1, 0, 2))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
        
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (1, 0, 2))
        b_o = tl.reshape(b_o, (BLOCK_T, H))
  
    tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))
    return


def chunk_local_cumsum_scalar_npu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    OPTIM_BLOCK_SIZE = triton.next_power_of_2((2 ** 18) // (H * chunk_size))
    block_indices = prepare_chunk_indices(cu_seqlens, chunk_size=OPTIM_BLOCK_SIZE) if cu_seqlens is not None else None
    num_blocks = len(block_indices) if cu_seqlens is not None else triton.cdiv(T, OPTIM_BLOCK_SIZE)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (num_blocks, B)

    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=block_indices,
        T=T,
        B=B,
        H=H,
        BLOCK_T = OPTIM_BLOCK_SIZE,
        CHUNK_SIZE =chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=8,
        num_stages=3,
    )
    return g

def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar_npu(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    elif len(g.shape) == 4:
        raise NotImplementedError(
            f"Unsupported input shape {g.shape} in ascend, "
            f"chunk_local_cumsum_vector is not implemented in NPU."
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++ solve_tril part ++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
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
    LARGE_BLOCK_T: tl.constexpr
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
    tl.device_print('i_t:', i_t)
    tl.device_print('base_t:', base_t)
    
    b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32) # (N_BLOCKS, 16, 16)
    for blkid in range(0, N_BLOCKS):
        row_start_o = base_t + blkid*16
        col_start_o = row_start_o % BT
        p_A_subrec16 = tl.make_block_ptr(
            A, (T, BT), (H*BT, 1), (row_start_o, col_start_o), (16, 16), (1, 0)
        )
        b_A_subrec16 = tl.load(p_A_subrec16, boundary_check=(0, 1)).to(tl.float32) # (16, 16)
        b_A = tl.insert_slice(
            ful=b_A,
            sub=b_A_subrec16[None, :, :], # (1, 16, 16)
            offsets=[blkid, 0, 0],
            sizes=[1, 16, 16],
            strides=[1, 1, 1]
        )
    
    
    # load multi 16x16 into UB
    local_ori_A = tl.trans(b_A, (1, 0, 2))
    local_ori_A = tl.reshape(local_ori_A, (16, 16*N_BLOCKS)) # (16, N_BLOCKS*16)
    
    tmp = tl.arange(0, 16).to(tl.float32)
    rows = tmp[:, None]
    cols = tmp[None, :]
    is_lower = (rows > cols).to(b_A.dtype)
    b_A = -b_A * is_lower
    
    o_i = tl.arange(0, 16)
    for i in range(1, 16):
        
        nblks_vec16 = -tl.extract_slice(local_ori_A, (i, 0), (1, 16*N_BLOCKS), (16*N_BLOCKS, 1))
        b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))
        
        dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
        dot_product = tl.sum(dot_tmp, 0)
        b_a = b_a + dot_product  # (N_BLOCKS, 16)
        
        row_mask = (o_i == i)  # (16,), True at position i
        update_mask = row_mask[None, :, None]  # (1, 16, 1)
        b_a_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
        b_A = tl.where(update_mask, b_a_expanded, b_A) # shape keeps (N_BLOCKS, 16, 16)
        
    
    on_diagonal = (rows == cols)
    b_A = tl.where(on_diagonal, b_A + 1.0, b_A)
    
    
    b_A = tl.reshape(b_A, (N_BLOCKS*16, 16))
    p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS*16, 16), (1, 0))
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1),)


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
    LARGE_BLOCK_T: tl.constexpr
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
        base_t += taskid * (LARGE_BLOCK_T//NTASKS)
        
        b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32) # (N_BLOCKS, 16, 16)
        for blkid in range(0, N_BLOCKS):
            row_start_o = base_t + blkid*16
            col_start_o = row_start_o % BT
            # using ptr with mask instead of tl.load(block_ptr)
            offs_rows_in_block = tl.arange(0, 16)
            offs_cols_in_block = tl.arange(0, 16)
            # strides (H*BT, 1)
            ptr_A_subrec16 = (A + row_start_o * H * BT + col_start_o + 
                              offs_rows_in_block[:, None] * H * BT + 
                              offs_cols_in_block[None, :])
            global_rows = row_start_o + offs_rows_in_block[:, None]
            global_cols = col_start_o + offs_cols_in_block[None, :]
            load_mask = (global_rows < T) & (global_cols < BT)
            b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask, other=0.0).to(tl.float32)
            b_A = tl.insert_slice(
                ful=b_A,
                sub=b_A_subrec16[None, :, :], # (1, 16, 16)
                offsets=[blkid, 0, 0],
                sizes=[1, 16, 16],
                strides=[1, 1, 1]
            )
        
        # load multi 16x16
        local_ori_A = tl.trans(b_A, (1, 0, 2))
        local_ori_A = tl.reshape(local_ori_A, (16, 16*N_BLOCKS)) # (16, N_BLOCKS*16)
        
        # change mask into matrix elementwise action
        tmp = tl.arange(0, 16).to(tl.float32)
        rows = tmp[:, None]
        cols = tmp[None, :]
        is_lower = (rows > cols).to(b_A.dtype)
        b_A = -b_A * is_lower
        
        for i in range(1, 16):
            
            nblks_vec16 = -tl.extract_slice(local_ori_A, (i, 0), (1, 16*N_BLOCKS), (16*N_BLOCKS, 1))
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
                strides=[1, 1, 1] 
            )

        on_diagonal = (rows == cols)
        b_A = tl.where(on_diagonal, b_A + 1.0, b_A)
        
        
        b_A = tl.reshape(b_A, (N_BLOCKS*16, 16))
        p_Ai = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS*16, 16), (1, 0))
        # using ptr with mask instead of tl.load(block_ptr)
        offs_rows_to_store = tl.arange(0, N_BLOCKS*16)
        offs_cols_to_store = tl.arange(0, 16)
        # strides (H*16, 1)
        p_Ai = (Ad + base_t * H * 16 + 0 +
                offs_rows_to_store[:, None] * H * 16 +
                offs_cols_to_store[None, :])
        global_store_rows = base_t + offs_rows_to_store[:, None]
        store_mask = global_store_rows < T
        tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=store_mask)


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


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
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
    tl.store(ptr_Ai, Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

    # ------------------ Store Ai_22_32 to (i_t*64+32, 32) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(ptr_Ai, Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

    # ------------------ Store Ai_21_32 to (i_t*64+32, 0) ------------------
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(ptr_Ai, Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)
    
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

    LARGE_BLOCK_T = 608*2
    # assert A.shape[1]%LARGE_BLOCK_T == 0 # or last N_BLOCKS have not enough block which leads to tl.arange failed
    
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T) if cu_seqlens is not None else None
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


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++ wy_fast part +++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel_npu_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):  
    T_max = T
    i_t_o, _ = tl.program_id(0), tl.program_id(1)
    for i_bh in range(H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T
        
        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = (A + (bos * H + i_h) * BT + 
                 offs_t_2d * (H * BT) + 
                 offs_bt * 1)
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        ptr_g = g + bos + i_h * T_max + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        ptr_beta = beta + bos + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            # --- load v (BTxBV) ---
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)
            # orig strides (H * V, 1)
            ptr_v = (v + (bos * H + i_h) * V +
                     offs_t_2d * (H * V) +
                     offs_v * 1)
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)
            
            b_vb = (b_v * b_beta[:, None])
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)
            ptr_u = (u + (bos * H + i_h) * V +
                     offs_t_2d * (H * V) +
                     offs_v * 1)
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            # orig strides (Hg * K, 1)
            ptr_k = (k + (bos * Hg + i_h // (H // Hg)) * K +
                     offs_t_2d * (Hg * K) +
                     offs_k * 1)
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)
            
            b_kb = (b_k * b_beta[:, None] * b_g[:, None])
            b_w = tl.dot(b_A, b_kb)
            ptr_w = (w + (bos * H + i_h) * K +
                     offs_t_2d * (H * K) +
                     offs_k * 1)
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 128
    BV = 128
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    # Pre-transpose tensors outside the kernel to ensure contiguous memory access within the kernel
    # avoiding scattered (non-contiguous) access that may lead to axis expansion
    beta = beta.transpose(1, 2).contiguous()
    g_cumsum = g_cumsum.transpose(1, 2).contiguous()
    recompute_w_u_fwd_kernel_npu_kernel[(NT, B)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )
    return w, u
