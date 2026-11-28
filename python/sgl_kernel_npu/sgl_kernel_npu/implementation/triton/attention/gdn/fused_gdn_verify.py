"""Ascend 950 (arch35) Cube-accelerated Triton kernels for GDN speculative verify and commit.

Provides three operators and an optional persistent ACLGraph-safe scratch container:

1. ``fused_gdn_verify``: Cube WY-chunk GDN verification from pre-convolved
   ``(mixed_qkv, a, b)`` tensors (``B <= 8`` single-stage kernel; ``B > 8`` and
   power-of-two ``K_VER`` two-stage multi-request R-packed kernel).
2. ``fused_conv1d_gdn_verify``: Fused strided-``qkvz`` causal Conv1D (``kernel=4``,
   SiLU) + in-place ``conv_states`` shift + Q/K L2Norm + GDN gating + ``V * beta``
   followed by Cube WY-chunk GDN verification.
3. ``fused_gdn_commit``: Single-launch all-layer Cube-accelerated in-place SSM state
   commit (with optional Radix Cache checkpoint write from entering state ``S_0``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["B"])
def _fused_conv1d_norm_gate_verify_kernel(
    QKVZ_ptr,  # [B * K_VER, QKVZ_DIM] bf16
    BA_ptr,  # [B * K_VER, 2 * NV] bf16
    CONV_W_ptr,  # [4, QKV_DIM] bf16
    STATE_ptr,  # [NUM_SLOTS, WIN_LEN, QKV_DIM] bf16 (WIN_LEN = K_VER + 2)
    CACHE_IDX_ptr,  # [B] int32
    QSL_ptr,  # [B + 1] int32 (dummy when HAS_QSL=False)
    A_LOG_ptr,  # [NV] fp32
    DT_BIAS_ptr,  # [NV] bf16/fp32
    Q_OUT_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    REC_K_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    VB_OUT_ptr,  # [MAX_BS, K_VER, NV, D] bf16 (conv1d(V) * beta)
    REC_G_ptr,  # [MAX_BS, K_VER, NV] fp32
    BETA_OUT_ptr,  # [MAX_BS, K_VER, NV] fp32
    B,
    QKVZ_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    WIN_LEN: tl.constexpr,
    K_VER: tl.constexpr,
    K_PAD: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    NV_PAD: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    H_BLK: tl.constexpr,
    HAS_QSL: tl.constexpr,
):
    """Fused strided-QKVZ verify Conv1D + Q/K L2Norm + GDN gating + V*beta.

    Matches causal conv1d speculative verify semantics (kernel_size=4, SiLU):
      - Initial 3-tap window = STATE[slot, WIN_LEN-3 .. WIN_LEN-1]
      - Post-step state = [old[WIN_LEN-2], old[WIN_LEN-1], x[0], ..., x[K_VER-1]]
    """
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    n_q = NK // H_BLK
    n_qk = 2 * n_q
    tasks_per_req = n_qk + (NV // H_BLK)
    total_tasks = B * tasks_per_req

    offs_h = tl.arange(0, H_BLK)
    offs_d = tl.arange(0, D)
    offs_hd = offs_h[:, None] * D + offs_d[None, :]
    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K_VER
    offs_nv = tl.arange(0, NV_PAD)
    mask_nv = offs_nv < NV
    a_log_all = tl.load(A_LOG_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)
    dt_b_all = tl.load(DT_BIAS_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)
    neg_exp_a_all = -tl.exp(a_log_all)

    for task in range(pid, total_tasks, n_progs):
        i_b = task // tasks_per_req
        sub = task - i_b * tasks_per_req
        slot = tl.load(CACHE_IDX_ptr + i_b).to(tl.int64)
        if HAS_QSL:
            q_start = tl.load(QSL_ptr + i_b)
            q_end = tl.load(QSL_ptr + i_b + 1)
            valid_slot = (slot >= 0) & (q_end > q_start)
        else:
            valid_slot = slot >= 0
        safe_slot = tl.where(valid_slot, slot, 0)
        s_base = safe_slot * (WIN_LEN * QKV_DIM)

        if sub == 0:
            toks = i_b * K_VER + offs_k
            mask_knv = mask_k[:, None] & mask_nv[None, :]
            ba_off = toks[:, None] * (2 * NV) + offs_nv[None, :]
            b_v = tl.load(BA_ptr + ba_off, mask=mask_knv, other=0.0).to(tl.float32)
            a_v = tl.load(BA_ptr + ba_off + NV, mask=mask_knv, other=0.0).to(tl.float32)
            sp = a_v + dt_b_all[None, :]
            g = tl.where(
                mask_knv,
                neg_exp_a_all[None, :]
                * tl.where(sp > 20.0, sp, tl.log(1.0 + tl.exp(sp))),
                0.0,
            )
            gc = tl.cumsum(g, axis=0)
            beta = tl.where(mask_knv, tl.sigmoid(b_v), 0.0)
            out_off = toks[:, None] * NV + offs_nv[None, :]
            tl.store(REC_G_ptr + out_off, gc, mask=mask_knv)
            tl.store(BETA_OUT_ptr + out_off, beta, mask=mask_knv)

        c_off = sub * (H_BLK * D) + offs_hd
        w0 = tl.load(CONV_W_ptr + c_off).to(tl.float32)
        w1 = tl.load(CONV_W_ptr + QKV_DIM + c_off).to(tl.float32)
        w2 = tl.load(CONV_W_ptr + 2 * QKV_DIM + c_off).to(tl.float32)
        w3 = tl.load(CONV_W_ptr + 3 * QKV_DIM + c_off).to(tl.float32)

        if valid_slot:
            s0_r = tl.load(STATE_ptr + s_base + (WIN_LEN - 3) * QKV_DIM + c_off)
            s1_r = tl.load(STATE_ptr + s_base + (WIN_LEN - 2) * QKV_DIM + c_off)
            s2_r = tl.load(STATE_ptr + s_base + (WIN_LEN - 1) * QKV_DIM + c_off)
            tl.store(STATE_ptr + s_base + c_off, s1_r)
            tl.store(STATE_ptr + s_base + QKV_DIM + c_off, s2_r)
            s0 = s0_r.to(tl.float32)
            s1 = s1_r.to(tl.float32)
            s2 = s2_r.to(tl.float32)
        else:
            s0 = tl.zeros((H_BLK, D), dtype=tl.float32)
            s1 = tl.zeros((H_BLK, D), dtype=tl.float32)
            s2 = tl.zeros((H_BLK, D), dtype=tl.float32)

        # Keep Q / K / V branches separate outside the static K_VER loop so each
        # tl.store uses a kernel-argument base pointer.
        if sub < n_q:
            q_off = sub * (H_BLK * D) + offs_hd
            for t in tl.static_range(K_VER):
                tok = i_b * K_VER + t
                x = tl.load(QKVZ_ptr + tok * QKVZ_DIM + c_off)
                if valid_slot:
                    tl.store(STATE_ptr + s_base + (2 + t) * QKV_DIM + c_off, x)
                xf = x.to(tl.float32)
                acc = s0 * w0 + s1 * w1 + s2 * w2 + xf * w3
                yf = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
                inv = tl.rsqrt(tl.sum(yf * yf, axis=1, keep_dims=True) + EPS)
                tl.store(Q_OUT_ptr + tok * (NK * D) + q_off, (yf * inv).to(tl.bfloat16))
                s0 = s1
                s1 = s2
                s2 = xf
        elif sub < n_qk:
            k_off = (sub - n_q) * (H_BLK * D) + offs_hd
            for t in tl.static_range(K_VER):
                tok = i_b * K_VER + t
                x = tl.load(QKVZ_ptr + tok * QKVZ_DIM + c_off)
                if valid_slot:
                    tl.store(STATE_ptr + s_base + (2 + t) * QKV_DIM + c_off, x)
                xf = x.to(tl.float32)
                acc = s0 * w0 + s1 * w1 + s2 * w2 + xf * w3
                yf = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
                inv = tl.rsqrt(tl.sum(yf * yf, axis=1, keep_dims=True) + EPS)
                tl.store(REC_K_ptr + tok * (NK * D) + k_off, (yf * inv).to(tl.bfloat16))
                s0 = s1
                s1 = s2
                s2 = xf
        else:
            v_h0 = (sub - n_qk) * H_BLK
            v_off = v_h0 * D + offs_hd
            for t in tl.static_range(K_VER):
                tok = i_b * K_VER + t
                x = tl.load(QKVZ_ptr + tok * QKVZ_DIM + c_off)
                if valid_slot:
                    tl.store(STATE_ptr + s_base + (2 + t) * QKV_DIM + c_off, x)
                xf = x.to(tl.float32)
                acc = s0 * w0 + s1 * w1 + s2 * w2 + xf * w3
                yf = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
                beta_h = tl.sigmoid(
                    tl.load(BA_ptr + tok * (2 * NV) + v_h0 + offs_h).to(tl.float32)
                )
                tl.store(
                    VB_OUT_ptr + tok * (NV * D) + v_off,
                    (yf * beta_h[:, None]).to(tl.bfloat16),
                )
                s0 = s1
                s1 = s2
                s2 = xf


@triton.jit(do_not_specialize=["B"])
def _s1a_norm_gate_vb_ab_kernel(
    MIXED_QKV_ptr,  # [B, K_VER, QKV_DIM] bf16
    A_ptr,  # [B, K_VER, NV] bf16/fp32
    B_ptr,  # [B, K_VER, NV] bf16/fp32
    A_LOG_ptr,  # [NV] fp32
    DT_BIAS_ptr,  # [NV] bf16/fp32
    Q_OUT_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    REC_K_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    VB_OUT_ptr,  # [MAX_BS, K_VER, NV, D] bf16
    REC_G_ptr,  # [MAX_BS, K_VER, NV] fp32
    BETA_OUT_ptr,  # [MAX_BS, K_VER, NV] fp32
    B,
    K_VER: tl.constexpr,
    K_PAD: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    NV_PAD: tl.constexpr,
    D: tl.constexpr,
    QKV_DIM: tl.constexpr,
    EPS: tl.constexpr,
    H_BLK: tl.constexpr,
):
    """56-AIV prep from pre-convolved (mixed_qkv, a, b) tensors."""
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    n_q = NK // H_BLK
    n_qk = 2 * n_q
    tasks_per_req = n_qk + (NV // H_BLK)
    total_tasks = B * tasks_per_req

    offs_h = tl.arange(0, H_BLK)
    offs_d = tl.arange(0, D)
    offs_hd = offs_h[:, None] * D + offs_d[None, :]
    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K_VER
    offs_nv = tl.arange(0, NV_PAD)
    mask_nv = offs_nv < NV
    neg_exp_a = -tl.exp(
        tl.load(A_LOG_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)
    )
    dt_b = tl.load(DT_BIAS_ptr + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)

    for task in range(pid, total_tasks, n_progs):
        i_b = task // tasks_per_req
        sub = task - i_b * tasks_per_req
        if sub == 0:
            toks = i_b * K_VER + offs_k
            mask_knv = mask_k[:, None] & mask_nv[None, :]
            ab_off = toks[:, None] * NV + offs_nv[None, :]
            a_v = tl.load(A_ptr + ab_off, mask=mask_knv, other=0.0).to(tl.float32)
            b_v = tl.load(B_ptr + ab_off, mask=mask_knv, other=0.0).to(tl.float32)
            sp = a_v + dt_b[None, :]
            g = tl.where(
                mask_knv,
                neg_exp_a[None, :] * tl.where(sp > 20.0, sp, tl.log(1.0 + tl.exp(sp))),
                0.0,
            )
            tl.store(REC_G_ptr + ab_off, tl.cumsum(g, axis=0), mask=mask_knv)
            tl.store(
                BETA_OUT_ptr + ab_off,
                tl.where(mask_knv, tl.sigmoid(b_v), 0.0),
                mask=mask_knv,
            )

        c_off = sub * (H_BLK * D) + offs_hd
        if sub < n_q:
            q_off = sub * (H_BLK * D) + offs_hd
            for t in tl.static_range(K_VER):
                tok = i_b * K_VER + t
                yf = tl.load(MIXED_QKV_ptr + tok * QKV_DIM + c_off).to(tl.float32)
                inv = tl.rsqrt(tl.sum(yf * yf, axis=1, keep_dims=True) + EPS)
                tl.store(Q_OUT_ptr + tok * (NK * D) + q_off, (yf * inv).to(tl.bfloat16))
        elif sub < n_qk:
            k_off = (sub - n_q) * (H_BLK * D) + offs_hd
            for t in tl.static_range(K_VER):
                tok = i_b * K_VER + t
                yf = tl.load(MIXED_QKV_ptr + tok * QKV_DIM + c_off).to(tl.float32)
                inv = tl.rsqrt(tl.sum(yf * yf, axis=1, keep_dims=True) + EPS)
                tl.store(REC_K_ptr + tok * (NK * D) + k_off, (yf * inv).to(tl.bfloat16))
        else:
            v_h0 = (sub - n_qk) * H_BLK
            v_off = v_h0 * D + offs_hd
            for t in tl.static_range(K_VER):
                tok = i_b * K_VER + t
                yf = tl.load(MIXED_QKV_ptr + tok * QKV_DIM + c_off).to(tl.float32)
                beta_h = tl.sigmoid(
                    tl.load(B_ptr + tok * NV + v_h0 + offs_h).to(tl.float32)
                )
                tl.store(
                    VB_OUT_ptr + tok * (NV * D) + v_off,
                    (yf * beta_h[:, None]).to(tl.bfloat16),
                )


@triton.jit(do_not_specialize=["scale"])
def _fused_gdn_verify_1stage_prepped_kernel(
    Q_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    K_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    VB_ptr,  # [MAX_BS, K_VER, NV, D] bf16
    REC_G_ptr,  # [MAX_BS, K_VER, NV] fp32
    BETA_ptr,  # [MAX_BS, K_VER, NV] fp32
    SSM_ptr,  # [NUM_SLOTS, NV, D, D] bf16 (read-only)
    CACHE_IDX_ptr,  # [B] int32
    O_ptr,  # [B, K_VER, NV, D] bf16
    REC_V_ptr,  # [MAX_BS, K_VER, NV, D] bf16
    scale,
    K_VER: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
):
    """Single-launch direct-grid ``(B * NV,)`` Cube GDN verify from prepped tensors."""
    pid = tl.program_id(0)
    i_b = pid // NV
    i_vh = pid % NV
    i_kh = i_vh // GQA
    slot = tl.load(CACHE_IDX_ptr + i_b).to(tl.int64)
    valid_slot = slot >= 0
    safe_slot = tl.where(valid_slot, slot, 0)

    offs_bt = tl.arange(0, 16)
    mask_t = offs_bt < K_VER
    offs_d = tl.arange(0, D)
    offs_dd = (offs_d[:, None] * D + offs_d[None, :]).to(tl.int64)
    rows = offs_bt[:, None]
    cols = offs_bt[None, :]
    strict_lower = (rows > cols) & mask_t[:, None] & mask_t[None, :]
    lower_eq = (rows >= cols) & mask_t[:, None] & mask_t[None, :]
    eye16 = tl.where((rows == cols) & mask_t[:, None], 1.0, 0.0).to(tl.float16)

    rk_off = (i_b * K_VER + offs_bt[:, None]) * (NK * D) + i_kh * D + offs_d[None, :]
    b_q = tl.load(Q_ptr + rk_off, mask=mask_t[:, None], other=0.0)
    b_k = tl.load(K_ptr + rk_off, mask=mask_t[:, None], other=0.0)
    b_k_t = tl.trans(b_k)
    b_kkt = tl.dot(b_k, b_k_t)
    b_qk = tl.dot(b_q, b_k_t)

    ab_off = (i_b * K_VER + offs_bt) * NV + i_vh
    gc = tl.load(REC_G_ptr + ab_off, mask=mask_t, other=0.0).to(tl.float32)
    beta_val = tl.load(BETA_ptr + ab_off, mask=mask_t, other=0.0).to(tl.float32)
    exp_gc = tl.exp(gc)

    eg = tl.where(
        lower_eq,
        tl.exp(tl.where(lower_eq, gc[:, None] - gc[None, :], -1e9)),
        0.0,
    )
    attn = (b_qk * eg).to(tl.bfloat16)

    l_bf = tl.where(strict_lower, b_kkt * beta_val[:, None] * eg, 0.0).to(tl.float16)
    l2 = tl.dot(l_bf, l_bf, out_dtype=tl.float16)
    t1 = tl.dot(eye16 - l_bf, eye16 + l2, out_dtype=tl.float16)
    if K_VER > 8:
        l4 = tl.dot(l2, l2, out_dtype=tl.float16)
        t2 = tl.dot(t1, eye16 + l4, out_dtype=tl.float16)
        l8 = tl.dot(l4, l4, out_dtype=tl.float16)
        t_inv = tl.dot(t2, eye16 + l8, out_dtype=tl.float16).to(tl.bfloat16)
    elif K_VER > 4:
        l4 = tl.dot(l2, l2, out_dtype=tl.float16)
        t_inv = tl.dot(t1, eye16 + l4, out_dtype=tl.float16).to(tl.bfloat16)
    else:
        t_inv = t1.to(tl.bfloat16)

    vb_off = (i_b * K_VER + offs_bt[:, None]) * (NV * D) + i_vh * D + offs_d[None, :]
    b_vb = tl.load(VB_ptr + vb_off, mask=mask_t[:, None], other=0.0)
    b_w = tl.dot(
        t_inv, (b_k.to(tl.float32) * (beta_val * exp_gc)[:, None]).to(tl.bfloat16)
    ).to(tl.bfloat16)
    b_u = tl.dot(t_inv, b_vb)

    state_off = (safe_slot * NV + i_vh) * (D * D) + offs_dd
    b_h0 = tl.load(SSM_ptr + state_off, mask=valid_slot, other=0.0)
    b_h0_t = tl.trans(b_h0)
    b_wh = tl.dot(b_w, b_h0_t)
    b_qh = tl.dot(b_q, b_h0_t)

    b_v_new = tl.where(mask_t[:, None], b_u - b_wh, 0.0).to(tl.bfloat16)
    tl.store(REC_V_ptr + vb_off, b_v_new, mask=mask_t[:, None])

    b_o = (b_qh * exp_gc[:, None] + tl.dot(attn, b_v_new)) * scale
    tl.store(O_ptr + vb_off, b_o.to(O_ptr.dtype.element_ty), mask=mask_t[:, None])


@triton.jit(do_not_specialize=["B"])
def _fused_gdn_verify_s1b_rpack_kernel(
    Q_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    K_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    VB_ptr,  # [MAX_BS, K_VER, NV, D] bf16
    REC_G_ptr,  # [MAX_BS, K_VER, NV] fp32
    BETA_ptr,  # [MAX_BS, K_VER, NV] fp32
    W_OUT_ptr,  # [MAX_BS, NV, 16, D] bf16 (rows K_VER..15 zero-initialized)
    U_OUT_ptr,  # [MAX_BS, NV, 16, D] bf16 (rows K_VER..15 zero-initialized)
    ATTN_OUT_ptr,  # [MAX_BS, NV, 16, 16] bf16 (padded rows/cols zero-initialized)
    B,
    K_VER: tl.constexpr,
    BT: tl.constexpr,
    R_PACK: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
):
    """Block-diagonal multi-request packed Stage 1b (``grid=(N_GRP * NK,)``)."""
    pid = tl.program_id(0)
    i_grp = pid // NK
    i_kh = pid % NK
    v0 = i_kh * GQA
    b0 = i_grp * R_PACK

    offs_bt = tl.arange(0, BT)
    req_in_grp = offs_bt // K_VER
    pos_in_req = offs_bt % K_VER
    tok_idx = b0 * K_VER + offs_bt
    mask_tok = (b0 + req_in_grp) < B
    offs_d = tl.arange(0, D)

    same_req = (
        (req_in_grp[:, None] == req_in_grp[None, :])
        & mask_tok[:, None]
        & mask_tok[None, :]
    )
    lower_eq = same_req & (pos_in_req[:, None] >= pos_in_req[None, :])
    strict_lower = same_req & (pos_in_req[:, None] > pos_in_req[None, :])
    eye_bt = tl.where(
        (offs_bt[:, None] == offs_bt[None, :]) & mask_tok[:, None], 1.0, 0.0
    ).to(tl.float16)

    rk_off = tok_idx[:, None] * (NK * D) + i_kh * D + offs_d[None, :]
    b_q = tl.load(Q_ptr + rk_off, mask=mask_tok[:, None], other=0.0)
    b_k = tl.load(K_ptr + rk_off, mask=mask_tok[:, None], other=0.0)
    b_k_t = tl.trans(b_k)
    b_kkt = tl.dot(b_k, b_k_t)
    b_qk = tl.dot(b_q, b_k_t)
    bk_f = b_k.to(tl.float32)

    for g_idx in range(GQA):
        vh = v0 + g_idx
        gc = tl.load(REC_G_ptr + tok_idx * NV + vh, mask=mask_tok, other=0.0).to(
            tl.float32
        )
        beta_val = tl.load(BETA_ptr + tok_idx * NV + vh, mask=mask_tok, other=0.0).to(
            tl.float32
        )
        exp_gc = tl.exp(gc)
        eg = tl.where(
            lower_eq,
            tl.exp(tl.where(lower_eq, gc[:, None] - gc[None, :], -1e9)),
            0.0,
        )
        attn_off = (
            ((b0 + req_in_grp[:, None]) * NV + vh) * 256
            + pos_in_req[:, None] * 16
            + pos_in_req[None, :]
        )
        tl.store(
            ATTN_OUT_ptr + attn_off,
            (b_qk * eg).to(tl.bfloat16),
            mask=same_req,
        )

        l_bf = tl.where(strict_lower, b_kkt * beta_val[:, None] * eg, 0.0).to(
            tl.float16
        )
        l2 = tl.dot(l_bf, l_bf, out_dtype=tl.float16)
        t1 = tl.dot(eye_bt - l_bf, eye_bt + l2, out_dtype=tl.float16)
        if K_VER > 8:
            l4 = tl.dot(l2, l2, out_dtype=tl.float16)
            t2 = tl.dot(t1, eye_bt + l4, out_dtype=tl.float16)
            l8 = tl.dot(l4, l4, out_dtype=tl.float16)
            t_inv = tl.dot(t2, eye_bt + l8, out_dtype=tl.float16).to(tl.bfloat16)
        elif K_VER > 4:
            l4 = tl.dot(l2, l2, out_dtype=tl.float16)
            t_inv = tl.dot(t1, eye_bt + l4, out_dtype=tl.float16).to(tl.bfloat16)
        else:
            t_inv = t1.to(tl.bfloat16)

        vb_off = tok_idx[:, None] * (NV * D) + vh * D + offs_d[None, :]
        b_vb = tl.load(VB_ptr + vb_off, mask=mask_tok[:, None], other=0.0)
        b_w = tl.dot(t_inv, (bk_f * (beta_val * exp_gc)[:, None]).to(tl.bfloat16)).to(
            tl.bfloat16
        )
        b_u = tl.dot(t_inv, b_vb)
        wu_off = (
            ((b0 + req_in_grp[:, None]) * NV + vh) * (16 * D)
            + pos_in_req[:, None] * D
            + offs_d[None, :]
        )
        tl.store(W_OUT_ptr + wu_off, b_w, mask=mask_tok[:, None])
        tl.store(
            U_OUT_ptr + wu_off,
            b_u.to(U_OUT_ptr.dtype.element_ty),
            mask=mask_tok[:, None],
        )


@triton.jit(do_not_specialize=["scale"])
def _fused_gdn_verify_s2_kernel(
    Q_ptr,  # [MAX_BS, K_VER, NK, D] bf16
    W_ptr,  # [MAX_BS, NV, 16, D] bf16 (zero-padded rows K_VER..15)
    U_ptr,  # [MAX_BS, NV, 16, D] bf16 (zero-padded rows K_VER..15)
    ATTN_ptr,  # [MAX_BS, NV, 16, 16] bf16 (zero-padded outside K_VER x K_VER)
    REC_G_ptr,  # [MAX_BS, K_VER, NV] fp32
    SSM_ptr,  # [NUM_SLOTS, NV, D, D] bf16 (read-only)
    CACHE_IDX_ptr,  # [B] int32
    O_ptr,  # [B, K_VER, NV, D] bf16
    REC_V_ptr,  # [MAX_BS, K_VER, NV, D] bf16
    scale,
    K_VER: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
):
    """Stage 2 (``grid=(B * NV,)``): Cube ``W@S0^T``, ``Q@S0^T``, ``ATTN@V_new``."""
    pid = tl.program_id(0)
    i_b = pid // NV
    i_vh = pid % NV
    i_kh = i_vh // GQA
    slot = tl.load(CACHE_IDX_ptr + i_b).to(tl.int64)
    valid_slot = slot >= 0
    safe_slot = tl.where(valid_slot, slot, 0)

    offs_16 = tl.arange(0, 16)
    mask_t = offs_16 < K_VER
    offs_d = tl.arange(0, D)
    offs_dd = (offs_d[:, None] * D + offs_d[None, :]).to(tl.int64)
    off_btd = offs_16[:, None] * D + offs_d[None, :]
    off_a = offs_16[:, None] * 16 + offs_16[None, :]

    b_w = tl.load(W_ptr + (i_b * NV + i_vh) * (16 * D) + off_btd)
    b_u = tl.load(U_ptr + (i_b * NV + i_vh) * (16 * D) + off_btd).to(tl.float32)
    b_attn = tl.load(ATTN_ptr + (i_b * NV + i_vh) * 256 + off_a)
    b_q = tl.load(
        Q_ptr
        + (i_b * K_VER + offs_16[:, None]) * (NK * D)
        + i_kh * D
        + offs_d[None, :],
        mask=mask_t[:, None],
        other=0.0,
    )
    b_g = tl.load(
        REC_G_ptr + (i_b * K_VER + offs_16) * NV + i_vh, mask=mask_t, other=0.0
    ).to(tl.float32)

    state_off = (safe_slot * NV + i_vh) * (D * D) + offs_dd
    b_h0 = tl.load(SSM_ptr + state_off, mask=valid_slot, other=0.0)
    b_h0_t = tl.trans(b_h0)
    b_wh = tl.dot(b_w, b_h0_t)
    b_qh = tl.dot(b_q, b_h0_t)

    b_v_new = tl.where(mask_t[:, None], b_u - b_wh, 0.0).to(tl.bfloat16)
    rec_v_off = (i_b * K_VER + offs_16[:, None]) * (NV * D) + i_vh * D + offs_d[None, :]
    tl.store(REC_V_ptr + rec_v_off, b_v_new, mask=mask_t[:, None])

    b_o = (b_qh * tl.exp(b_g)[:, None] + tl.dot(b_attn, b_v_new)) * scale
    tl.store(O_ptr + rec_v_off, b_o.to(O_ptr.dtype.element_ty), mask=mask_t[:, None])


@triton.jit(do_not_specialize=["B", "NUM_LAYERS"])
def _fused_gdn_commit_kernel(
    SSM_ALL_ptr,  # [NUM_LAYERS, NUM_SLOTS, NV, D, D] bf16 (updated in place)
    REC_K_ALL_ptr,  # [NUM_LAYERS, MAX_BS, K_VER, NK, D] bf16
    REC_V_ALL_ptr,  # [NUM_LAYERS, MAX_BS, K_VER, NV, D] bf16
    REC_G_ALL_ptr,  # [NUM_LAYERS, MAX_BS, K_VER, NV] fp32
    DST_SLOTS_ptr,  # [B] int64
    LAST_STEPS_ptr,  # [B] int64 (0..K_VER-1)
    TRACK_SLOTS_ptr,  # [B] int64 (dummy when HAS_TRACK=False)
    TRACK_STEPS_ptr,  # [B] int64 (-1 when no track for this req)
    B,
    NUM_LAYERS,
    ssm_layer_stride,
    ssm_slot_stride,
    rec_k_layer_stride,
    rec_k_b_stride,
    rec_v_layer_stride,
    rec_v_b_stride,
    rec_g_layer_stride,
    rec_g_b_stride,
    K_VER: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    """Single-launch all-layer Cube-accelerated in-place SSM commit kernel."""
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    total_tasks = NUM_LAYERS * B * NV

    offs_bt = tl.arange(0, 16)
    mask_t = offs_bt < K_VER
    offs_d = tl.arange(0, D)
    offs_dd = (offs_d[:, None] * D + offs_d[None, :]).to(tl.int64)

    for task in range(pid, total_tasks, n_progs):
        i_l = task // (B * NV)
        rem = task - i_l * (B * NV)
        i_b = rem // NV
        i_vh = rem - i_b * NV
        i_kh = i_vh // GQA

        dst_slot = tl.load(DST_SLOTS_ptr + i_b).to(tl.int64)
        last_step = tl.load(LAST_STEPS_ptr + i_b).to(tl.int64)
        if dst_slot >= 0 and last_step >= 0 and last_step < K_VER:
            ssm_l_base = tl.cast(i_l, tl.int64) * ssm_layer_stride
            state_off = (
                ssm_l_base + dst_slot * ssm_slot_stride + i_vh * (D * D) + offs_dd
            )

            rk_base = (
                tl.cast(i_l, tl.int64) * rec_k_layer_stride
                + tl.cast(i_b, tl.int64) * rec_k_b_stride
                + offs_bt[:, None] * (NK * D)
                + i_kh * D
                + offs_d[None, :]
            )
            b_k = tl.load(REC_K_ALL_ptr + rk_base, mask=mask_t[:, None], other=0.0)

            rv_off = (
                tl.cast(i_l, tl.int64) * rec_v_layer_stride
                + tl.cast(i_b, tl.int64) * rec_v_b_stride
                + offs_bt[:, None] * (NV * D)
                + i_vh * D
                + offs_d[None, :]
            )
            b_v = tl.load(REC_V_ALL_ptr + rv_off, mask=mask_t[:, None], other=0.0).to(
                tl.float32
            )

            rg_lb_base = (
                tl.cast(i_l, tl.int64) * rec_g_layer_stride
                + tl.cast(i_b, tl.int64) * rec_g_b_stride
            )
            b_g = tl.load(
                REC_G_ALL_ptr + rg_lb_base + offs_bt * NV + i_vh,
                mask=mask_t,
                other=0.0,
            ).to(tl.float32)

            g_last = tl.load(REC_G_ALL_ptr + rg_lb_base + last_step * NV + i_vh).to(
                tl.float32
            )
            m_acc = mask_t & (offs_bt <= last_step)
            dec_acc = tl.where(m_acc, tl.exp(tl.where(m_acc, g_last - b_g, -1e9)), 0.0)
            v_acc = (b_v * dec_acc[:, None]).to(tl.bfloat16)
            upd_acc = tl.dot(tl.trans(v_acc), b_k)

            b_h0 = tl.load(SSM_ALL_ptr + state_off).to(tl.float32)

            if HAS_TRACK:
                track_slot = tl.load(TRACK_SLOTS_ptr + i_b).to(tl.int64)
                track_step = tl.load(TRACK_STEPS_ptr + i_b).to(tl.int64)
                if track_slot >= 0 and track_step >= 0 and track_step < K_VER:
                    g_trk = tl.load(
                        REC_G_ALL_ptr + rg_lb_base + track_step * NV + i_vh
                    ).to(tl.float32)
                    m_trk = mask_t & (offs_bt <= track_step)
                    dec_trk = tl.where(
                        m_trk, tl.exp(tl.where(m_trk, g_trk - b_g, -1e9)), 0.0
                    )
                    v_trk = (b_v * dec_trk[:, None]).to(tl.bfloat16)
                    h_trk = b_h0 * tl.exp(g_trk) + tl.dot(tl.trans(v_trk), b_k)
                    trk_off = (
                        ssm_l_base
                        + track_slot * ssm_slot_stride
                        + i_vh * (D * D)
                        + offs_dd
                    )
                    tl.store(
                        SSM_ALL_ptr + trk_off, h_trk.to(SSM_ALL_ptr.dtype.element_ty)
                    )

            h_new = b_h0 * tl.exp(g_last) + upd_acc
            tl.store(SSM_ALL_ptr + state_off, h_new.to(SSM_ALL_ptr.dtype.element_ty))


class FusedGDNVerifyWorkspace:
    """Persistent static scratch workspace for ACLGraph-safe fused Cube GDN verify & commit."""

    def __init__(self) -> None:
        self.rec_k_all: Optional[torch.Tensor] = None
        self.rec_v_all: Optional[torch.Tensor] = None
        self.rec_g_all: Optional[torch.Tensor] = None
        self.q_tmp: Optional[torch.Tensor] = None
        self.vb_tmp: Optional[torch.Tensor] = None
        self.beta_tmp: Optional[torch.Tensor] = None
        self.w_tmp: Optional[torch.Tensor] = None
        self.u_tmp: Optional[torch.Tensor] = None
        self.a_tmp: Optional[torch.Tensor] = None
        self.max_bs: int = 0
        self.k_ver: int = 0
        self.num_layers: int = 0
        self.nk: int = 0
        self.nv: int = 0
        self.head_dim: int = 0

    def ensure_allocated(
        self,
        *,
        num_layers: int,
        min_bs: int,
        k_ver: int,
        nk: int,
        nv: int,
        head_dim: int,
        device: torch.device,
    ) -> bool:
        """Ensure static workspace covers ``(num_layers, min_bs, k_ver)``.

        Returns ``False`` if reallocation is required while a graph capture is active.
        """
        if (
            self.rec_k_all is not None
            and self.rec_k_all.device == device
            and self.num_layers == num_layers
            and self.max_bs >= min_bs
            and self.k_ver == k_ver
            and self.nk == nk
            and self.nv == nv
            and self.head_dim == head_dim
        ):
            return True

        if torch.get_device_module().is_current_stream_capturing():
            return False

        alloc_bs = max(min_bs, self.max_bs, 64)
        self.rec_k_all = torch.empty(
            (num_layers, alloc_bs, k_ver, nk, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.rec_v_all = torch.empty(
            (num_layers, alloc_bs, k_ver, nv, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.rec_g_all = torch.empty(
            (num_layers, alloc_bs, k_ver, nv),
            dtype=torch.float32,
            device=device,
        )
        self.q_tmp = torch.empty(
            (alloc_bs, k_ver, nk, head_dim), dtype=torch.bfloat16, device=device
        )
        self.vb_tmp = torch.empty(
            (alloc_bs, k_ver, nv, head_dim), dtype=torch.bfloat16, device=device
        )
        self.beta_tmp = torch.empty(
            (alloc_bs, k_ver, nv), dtype=torch.float32, device=device
        )
        self.w_tmp = torch.zeros(
            (alloc_bs, nv, 16, head_dim), dtype=torch.bfloat16, device=device
        )
        self.u_tmp = torch.zeros(
            (alloc_bs, nv, 16, head_dim), dtype=torch.bfloat16, device=device
        )
        self.a_tmp = torch.zeros(
            (alloc_bs, nv, 16, 16), dtype=torch.bfloat16, device=device
        )
        self.max_bs = alloc_bs
        self.k_ver = k_ver
        self.num_layers = num_layers
        self.nk = nk
        self.nv = nv
        self.head_dim = head_dim
        return True


def _choose_h_blk(nk: int, nv: int, batch_size: int) -> int:
    if batch_size >= 16 and nk % 16 == 0 and nv % 16 == 0:
        return 16
    if batch_size >= 8 and nk % 8 == 0 and nv % 8 == 0:
        return 8
    for cand in (4, 8, 16, 2, 1):
        if nk % cand == 0 and nv % cand == 0:
            return cand
    return 1


def _validate_verify_common(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    batch_size: int,
    k_ver: int,
    nk: int,
    nv: int,
    head_dim: int,
) -> None:
    if head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    if not (2 <= k_ver <= 16):
        raise ValueError(f"k_ver must be in [2, 16], got {k_ver}")
    if nk <= 0 or nv <= 0 or nv % nk != 0:
        raise ValueError(
            f"num_k_heads ({nk}) must be positive and divide num_v_heads ({nv})"
        )
    if ssm_states.dtype != torch.bfloat16 or not ssm_states.is_contiguous():
        raise ValueError("ssm_states must be contiguous bfloat16")
    if ssm_states.ndim != 4 or ssm_states.shape[1:] != (nv, head_dim, head_dim):
        raise ValueError(
            f"ssm_states must have shape [num_slots, {nv}, {head_dim}, {head_dim}], "
            f"got {tuple(ssm_states.shape)}"
        )
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("A_log and dt_bias must be contiguous")
    if cache_indices.ndim != 1 or cache_indices.shape[0] < batch_size:
        raise ValueError(
            f"cache_indices must be 1D with length >= {batch_size}, "
            f"got {tuple(cache_indices.shape)}"
        )


def _resolve_buffers(
    *,
    B: int,
    K_VER: int,
    nk: int,
    nv: int,
    D: int,
    device: torch.device,
    workspace: Optional[FusedGDNVerifyWorkspace],
    linear_layer_idx: int,
    need_2stage_scratch: bool,
):
    if workspace is not None:
        if (
            workspace.rec_k_all is None
            or workspace.max_bs < B
            or workspace.k_ver != K_VER
            or workspace.nk != nk
            or workspace.nv != nv
            or workspace.head_dim != D
        ):
            raise ValueError(
                "workspace is uninitialized or incompatible with verify shape; "
                "call workspace.ensure_allocated(...) first"
            )
        return (
            workspace.q_tmp,
            workspace.vb_tmp,
            workspace.beta_tmp,
            workspace.rec_k_all[linear_layer_idx],
            workspace.rec_v_all[linear_layer_idx],
            workspace.rec_g_all[linear_layer_idx],
            workspace.w_tmp,
            workspace.u_tmp,
            workspace.a_tmp,
        )

    q_tmp = torch.empty((B, K_VER, nk, D), dtype=torch.bfloat16, device=device)
    vb_tmp = torch.empty((B, K_VER, nv, D), dtype=torch.bfloat16, device=device)
    beta_tmp = torch.empty((B, K_VER, nv), dtype=torch.float32, device=device)
    rec_k = torch.empty((B, K_VER, nk, D), dtype=torch.bfloat16, device=device)
    rec_v = torch.empty((B, K_VER, nv, D), dtype=torch.bfloat16, device=device)
    rec_g = torch.empty((B, K_VER, nv), dtype=torch.float32, device=device)
    if need_2stage_scratch:
        w_tmp = torch.zeros((B, nv, 16, D), dtype=torch.bfloat16, device=device)
        u_tmp = torch.zeros((B, nv, 16, D), dtype=torch.bfloat16, device=device)
        a_tmp = torch.zeros((B, nv, 16, 16), dtype=torch.bfloat16, device=device)
    else:
        w_tmp = u_tmp = a_tmp = None
    return q_tmp, vb_tmp, beta_tmp, rec_k, rec_v, rec_g, w_tmp, u_tmp, a_tmp


def _run_gdn_verify_from_prepped(
    *,
    B: int,
    K_VER: int,
    nk: int,
    nv: int,
    D: int,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    q_tmp: torch.Tensor,
    vb_tmp: torch.Tensor,
    beta_tmp: torch.Tensor,
    rec_k_layer: torch.Tensor,
    rec_v_layer: torch.Tensor,
    rec_g_layer: torch.Tensor,
    w_tmp: Optional[torch.Tensor],
    u_tmp: Optional[torch.Tensor],
    a_tmp: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    GQA = nv // nk
    scale = D**-0.5
    out = torch.empty((B, K_VER, nv, D), dtype=out_dtype, device=ssm_states.device)

    if B <= 8 or (16 % K_VER != 0):
        _fused_gdn_verify_1stage_prepped_kernel[(B * nv,)](
            q_tmp,
            rec_k_layer,
            vb_tmp,
            rec_g_layer,
            beta_tmp,
            ssm_states,
            cache_indices,
            out,
            rec_v_layer,
            scale,
            K_VER=K_VER,
            NK=nk,
            NV=nv,
            GQA=GQA,
            D=D,
        )
    else:
        bt = 32 if (32 % K_VER == 0 and B >= (32 // K_VER)) else 16
        r_pack = bt // K_VER
        n_grp = (B + r_pack - 1) // r_pack
        _fused_gdn_verify_s1b_rpack_kernel[(n_grp * nk,)](
            q_tmp,
            rec_k_layer,
            vb_tmp,
            rec_g_layer,
            beta_tmp,
            w_tmp,
            u_tmp,
            a_tmp,
            B,
            K_VER=K_VER,
            BT=bt,
            R_PACK=r_pack,
            NK=nk,
            NV=nv,
            GQA=GQA,
            D=D,
        )
        _fused_gdn_verify_s2_kernel[(B * nv,)](
            q_tmp,
            w_tmp,
            u_tmp,
            a_tmp,
            rec_g_layer,
            ssm_states,
            cache_indices,
            out,
            rec_v_layer,
            scale,
            K_VER=K_VER,
            NK=nk,
            NV=nv,
            GQA=GQA,
            D=D,
        )
    return out


def fused_gdn_verify(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int = 128,
    eps: float = 1e-6,
    workspace: Optional[FusedGDNVerifyWorkspace] = None,
    linear_layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cube GDN speculative verify from pre-convolved ``(mixed_qkv, a, b)``."""
    if (
        mixed_qkv.ndim != 3
        or mixed_qkv.dtype != torch.bfloat16
        or not mixed_qkv.is_contiguous()
    ):
        raise ValueError("mixed_qkv must be contiguous 3D bfloat16 [B, K_VER, QKV_DIM]")
    B, K_VER, QKV_DIM = mixed_qkv.shape
    expected_qkv = (2 * num_k_heads + num_v_heads) * head_dim
    if QKV_DIM != expected_qkv:
        raise ValueError(f"mixed_qkv last dim must be {expected_qkv}, got {QKV_DIM}")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    if a.shape != (B, K_VER, num_v_heads) or b.shape != (B, K_VER, num_v_heads):
        raise ValueError(
            f"a and b must have shape {(B, K_VER, num_v_heads)}, "
            f"got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    _validate_verify_common(
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        B,
        K_VER,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    if B == 0:
        dev = mixed_qkv.device
        return (
            torch.empty(
                (0, K_VER, num_v_heads, head_dim), dtype=mixed_qkv.dtype, device=dev
            ),
            torch.empty(
                (0, K_VER, num_k_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty(
                (0, K_VER, num_v_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty((0, K_VER, num_v_heads), dtype=torch.float32, device=dev),
        )

    need_2stage = B > 8 and (16 % K_VER == 0)
    (
        q_tmp,
        vb_tmp,
        beta_tmp,
        rec_k_layer,
        rec_v_layer,
        rec_g_layer,
        w_tmp,
        u_tmp,
        a_tmp,
    ) = _resolve_buffers(
        B=B,
        K_VER=K_VER,
        nk=num_k_heads,
        nv=num_v_heads,
        D=head_dim,
        device=mixed_qkv.device,
        workspace=workspace,
        linear_layer_idx=linear_layer_idx,
        need_2stage_scratch=need_2stage,
    )

    h_blk = _choose_h_blk(num_k_heads, num_v_heads, B)
    tasks_per_req = (2 * num_k_heads + num_v_heads) // h_blk
    grid = (min(56, B * tasks_per_req),)
    _s1a_norm_gate_vb_ab_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        q_tmp,
        rec_k_layer,
        vb_tmp,
        rec_g_layer,
        beta_tmp,
        B,
        K_VER=K_VER,
        K_PAD=triton.next_power_of_2(K_VER),
        NK=num_k_heads,
        NV=num_v_heads,
        NV_PAD=triton.next_power_of_2(num_v_heads),
        D=head_dim,
        QKV_DIM=QKV_DIM,
        EPS=eps,
        H_BLK=h_blk,
    )
    out = _run_gdn_verify_from_prepped(
        B=B,
        K_VER=K_VER,
        nk=num_k_heads,
        nv=num_v_heads,
        D=head_dim,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        q_tmp=q_tmp,
        vb_tmp=vb_tmp,
        beta_tmp=beta_tmp,
        rec_k_layer=rec_k_layer,
        rec_v_layer=rec_v_layer,
        rec_g_layer=rec_g_layer,
        w_tmp=w_tmp,
        u_tmp=u_tmp,
        a_tmp=a_tmp,
        out_dtype=mixed_qkv.dtype,
    )
    return out, rec_k_layer[:B], rec_v_layer[:B], rec_g_layer[:B]


def fused_conv1d_gdn_verify(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_weights_t: torch.Tensor,
    conv_states: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    batch_size: int,
    k_ver: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int = 128,
    eps: float = 1e-6,
    query_start_loc: Optional[torch.Tensor] = None,
    workspace: Optional[FusedGDNVerifyWorkspace] = None,
    linear_layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused strided-QKVZ Conv1D + L2Norm + Gating + Cube WY GDN speculative verify."""
    qkv_dim = (2 * num_k_heads + num_v_heads) * head_dim
    if qkvz.ndim != 2 or qkvz.dtype != torch.bfloat16 or not qkvz.is_contiguous():
        raise ValueError("qkvz must be contiguous 2D bfloat16 [B * k_ver, qkvz_dim]")
    if qkvz.shape[0] != batch_size * k_ver or qkvz.shape[1] < qkv_dim:
        raise ValueError(
            f"qkvz must have shape [{batch_size * k_ver}, >={qkv_dim}], "
            f"got {tuple(qkvz.shape)}"
        )
    if ba.ndim != 2 or ba.dtype != torch.bfloat16 or not ba.is_contiguous():
        raise ValueError("ba must be contiguous 2D bfloat16 [B * k_ver, 2 * NV]")
    if ba.shape != (batch_size * k_ver, 2 * num_v_heads):
        raise ValueError(
            f"ba must have shape {(batch_size * k_ver, 2 * num_v_heads)}, "
            f"got {tuple(ba.shape)}"
        )
    if (
        conv_weights_t.ndim != 2
        or conv_weights_t.shape != (4, qkv_dim)
        or not conv_weights_t.is_contiguous()
    ):
        raise ValueError(
            f"conv_weights_t must be contiguous [4, {qkv_dim}], "
            f"got {tuple(conv_weights_t.shape)}"
        )
    if (
        conv_states.ndim != 3
        or conv_states.dtype != torch.bfloat16
        or not conv_states.is_contiguous()
        or conv_states.shape[1:] != (k_ver + 2, qkv_dim)
    ):
        raise ValueError(
            f"conv_states must be contiguous bfloat16 [num_slots, {k_ver + 2}, {qkv_dim}], "
            f"got {tuple(conv_states.shape)}"
        )
    _validate_verify_common(
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        batch_size,
        k_ver,
        num_k_heads,
        num_v_heads,
        head_dim,
    )
    if batch_size == 0:
        dev = qkvz.device
        return (
            torch.empty(
                (0, k_ver, num_v_heads, head_dim), dtype=qkvz.dtype, device=dev
            ),
            torch.empty(
                (0, k_ver, num_k_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty(
                (0, k_ver, num_v_heads, head_dim), dtype=torch.bfloat16, device=dev
            ),
            torch.empty((0, k_ver, num_v_heads), dtype=torch.float32, device=dev),
        )

    need_2stage = batch_size > 8 and (16 % k_ver == 0)
    (
        q_tmp,
        vb_tmp,
        beta_tmp,
        rec_k_layer,
        rec_v_layer,
        rec_g_layer,
        w_tmp,
        u_tmp,
        a_tmp,
    ) = _resolve_buffers(
        B=batch_size,
        K_VER=k_ver,
        nk=num_k_heads,
        nv=num_v_heads,
        D=head_dim,
        device=qkvz.device,
        workspace=workspace,
        linear_layer_idx=linear_layer_idx,
        need_2stage_scratch=need_2stage,
    )

    h_blk = _choose_h_blk(num_k_heads, num_v_heads, batch_size)
    tasks_per_req = (2 * num_k_heads + num_v_heads) // h_blk
    grid = (min(56, batch_size * tasks_per_req),)
    has_qsl = (
        query_start_loc is not None
        and query_start_loc.ndim == 1
        and query_start_loc.shape[0] >= batch_size + 1
    )
    qsl_arg = query_start_loc if has_qsl else cache_indices

    _fused_conv1d_norm_gate_verify_kernel[grid](
        qkvz,
        ba,
        conv_weights_t,
        conv_states,
        cache_indices,
        qsl_arg,
        A_log,
        dt_bias,
        q_tmp,
        rec_k_layer,
        vb_tmp,
        rec_g_layer,
        beta_tmp,
        batch_size,
        QKVZ_DIM=qkvz.shape[-1],
        QKV_DIM=qkv_dim,
        WIN_LEN=conv_states.shape[1],
        K_VER=k_ver,
        K_PAD=triton.next_power_of_2(k_ver),
        NK=num_k_heads,
        NV=num_v_heads,
        NV_PAD=triton.next_power_of_2(num_v_heads),
        D=head_dim,
        EPS=eps,
        H_BLK=h_blk,
        HAS_QSL=has_qsl,
    )
    out = _run_gdn_verify_from_prepped(
        B=batch_size,
        K_VER=k_ver,
        nk=num_k_heads,
        nv=num_v_heads,
        D=head_dim,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        q_tmp=q_tmp,
        vb_tmp=vb_tmp,
        beta_tmp=beta_tmp,
        rec_k_layer=rec_k_layer,
        rec_v_layer=rec_v_layer,
        rec_g_layer=rec_g_layer,
        w_tmp=w_tmp,
        u_tmp=u_tmp,
        a_tmp=a_tmp,
        out_dtype=qkvz.dtype,
    )
    return (
        out,
        rec_k_layer[:batch_size],
        rec_v_layer[:batch_size],
        rec_g_layer[:batch_size],
    )


def fused_gdn_commit(
    ssm_states_all: torch.Tensor,
    rec_k_all: torch.Tensor,
    rec_v_all: torch.Tensor,
    rec_g_all: torch.Tensor,
    dst_slots: torch.Tensor,
    last_steps: torch.Tensor,
    track_slots: Optional[torch.Tensor] = None,
    track_steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-launch all-layer Cube-accelerated in-place SSM state commit."""
    if ssm_states_all.ndim != 5 or ssm_states_all.dtype != torch.bfloat16:
        raise ValueError(
            "ssm_states_all must be 5D bfloat16 [NUM_LAYERS, NUM_SLOTS, NV, D, D]"
        )
    if rec_k_all.ndim != 5 or rec_v_all.ndim != 5 or rec_g_all.ndim != 4:
        raise ValueError("rec_k_all, rec_v_all must be 5D and rec_g_all must be 4D")
    num_layers, _, nv, d1, d2 = ssm_states_all.shape
    if d1 != 128 or d2 != 128:
        raise ValueError(f"SSM state head_dim must be 128x128, got {d1}x{d2}")
    k_ver = rec_k_all.shape[2]
    nk = rec_k_all.shape[3]
    if not (2 <= k_ver <= 16) or nk <= 0 or nv % nk != 0:
        raise ValueError(
            f"Unsupported verify commit shape: k_ver={k_ver}, nk={nk}, nv={nv}"
        )
    if (
        dst_slots.ndim != 1
        or last_steps.ndim != 1
        or dst_slots.shape != last_steps.shape
    ):
        raise ValueError("dst_slots and last_steps must be 1D tensors of equal length")
    B = dst_slots.shape[0]
    if B == 0:
        return ssm_states_all
    if (
        rec_k_all.shape[0] != num_layers
        or rec_v_all.shape[0] != num_layers
        or rec_g_all.shape[0] != num_layers
    ):
        raise ValueError(
            "Layer count mismatch between ssm_states_all and commit records"
        )
    if rec_k_all.shape[1] < B or rec_v_all.shape[1] < B or rec_g_all.shape[1] < B:
        raise ValueError("Batch extent of commit records must be >= len(dst_slots)")
    has_track = track_slots is not None and track_steps is not None
    if has_track and (track_slots.shape != (B,) or track_steps.shape != (B,)):
        raise ValueError("track_slots and track_steps must have shape [B]")

    dst_c = dst_slots.contiguous()
    last_c = last_steps.contiguous()
    trk_slots_c = track_slots.contiguous() if has_track else dst_c
    trk_steps_c = track_steps.contiguous() if has_track else dst_c

    total_tasks = num_layers * B * nv
    grid = (min(total_tasks, 28),)
    _fused_gdn_commit_kernel[grid](
        ssm_states_all,
        rec_k_all,
        rec_v_all,
        rec_g_all,
        dst_c,
        last_c,
        trk_slots_c,
        trk_steps_c,
        B,
        num_layers,
        ssm_states_all.stride(0),
        ssm_states_all.stride(1),
        rec_k_all.stride(0),
        rec_k_all.stride(1),
        rec_v_all.stride(0),
        rec_v_all.stride(1),
        rec_g_all.stride(0),
        rec_g_all.stride(1),
        K_VER=k_ver,
        NK=nk,
        NV=nv,
        GQA=nv // nk,
        D=d1,
        HAS_TRACK=has_track,
    )
    return ssm_states_all
