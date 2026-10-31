# Adapted and Merge from
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, By Triton_Ascend & sglang_ascend

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .gdn_utils import prepare_chunk_indices


def _validate_inputs(k, v, beta, g_cumsum, A, cu_seqlens):
    assert k.ndim == v.ndim == A.ndim == 4, "k, v and A must be 4-D"
    B, T, Hg, K = k.shape
    assert all(n > 0 for n in k.shape + v.shape + A.shape), "Inputs must be nonempty"
    H = v.shape[2]
    assert v.shape[:2] == (B, T), "k and v must share batch and sequence dimensions"
    assert H % Hg == 0, "Value heads must be a multiple of key heads"
    assert A.shape[:3] == (B, T, H), "A must have shape [B, T, H, C]"
    assert beta.shape == g_cumsum.shape == (B, T, H), "Gates must have shape [B, T, H]"
    assert A.shape[-1] in (16, 32, 64), "Triton chunk size must be 16, 32 or 64"
    assert k.device.type == "npu", "Triton requires NPU input"
    tensors = (k, v, beta, g_cumsum, A)
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert all(t.dtype in dtypes for t in tensors), "Inputs must be FP16, BF16 or FP32"
    assert all(t.device == k.device for t in tensors), "Inputs must share a device"
    assert all(t.is_contiguous() for t in (k, v, A)), "k, v and A must be contiguous"
    if cu_seqlens is not None:
        assert B == 1, "Packed input requires B=1"
        assert (
            cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2
        ), "cu_seqlens must be 1-D with at least two entries"
        assert cu_seqlens.dtype in (
            torch.int32,
            torch.int64,
        ), "cu_seqlens must be int32 or int64"
        assert (
            cu_seqlens.device == k.device and cu_seqlens.is_contiguous()
        ), "cu_seqlens must be contiguous on k's device"


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
    i_t_o, i_b = tl.program_id(0), tl.program_id(1)
    for i_h in range(H):
        i_t = i_t_o
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
        ptr_A = A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt * 1
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        # Gates are [B, H, T] after the wrapper transpose; k/v/A remain
        # [B, T, H, ...]. Packed sequences share batch zero.
        gate_bos = bos if IS_VARLEN else i_b * H * T_max
        ptr_g = g + gate_bos + i_h * T_max + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        ptr_beta = beta + gate_bos + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            # --- load v (BTxBV) ---
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)
            # orig strides (H * V, 1)
            ptr_v = v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)

            b_vb = b_v * b_beta[:, None]
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)
            ptr_u = u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            # orig strides (Hg * K, 1)
            ptr_k = (
                k
                + (bos * Hg + i_h // (H // Hg)) * K
                + offs_t_2d * (Hg * K)
                + offs_k * 1
            )
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)

            b_kb = b_k * b_beta[:, None] * b_g[:, None]
            b_w = tl.dot(b_A, b_kb)
            ptr_w = w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k * 1
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(k, v, beta, g_cumsum, A, cu_seqlens)
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
