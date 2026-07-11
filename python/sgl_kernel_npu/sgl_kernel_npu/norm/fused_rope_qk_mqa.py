import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_qk_mqa_kernel_opt(
    query_ptr,  # [T, Hq, D]
    key_ptr,  # [T, Hk, D]
    cos_sin_ptr,  # [max_pos, D_ROPE]
    out_q_ptr,
    out_k_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_ct,
    stride_cd,
    stride_oqt,
    stride_oqh,
    stride_oqd,
    stride_okt,
    stride_okh,
    stride_okd,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_ROPE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    pid_t = tl.program_id(0)

    # -------- rotary indices
    d = tl.arange(0, D_ROPE // 2)
    if IS_NEOX_STYLE:
        idx_even = d
        idx_odd = d + D_ROPE // 2
    else:
        idx_even = d * 2
        idx_odd = d * 2 + 1

    # cos / sin (shared across all heads for this position)
    cos = tl.load(cos_sin_ptr + pid_t * stride_ct + d * stride_cd)
    sin = tl.load(cos_sin_ptr + pid_t * stride_ct + (d + D_ROPE // 2) * stride_cd)

    # ================= Q (all Hq heads at once) =================
    head_offs = tl.arange(0, Hq)
    q_base = query_ptr + pid_t * stride_qt

    q1 = tl.load(
        q_base + head_offs[:, None] * stride_qh + idx_even[None, :] * stride_qd
    )
    q2 = tl.load(q_base + head_offs[:, None] * stride_qh + idx_odd[None, :] * stride_qd)

    q_out1 = (q1 * cos[None, :]) - (q2 * sin[None, :])
    q_out2 = (q1 * sin[None, :]) + (q2 * cos[None, :])

    oq_base = out_q_ptr + pid_t * stride_oqt
    tl.store(
        oq_base + head_offs[:, None] * stride_oqh + idx_even[None, :] * stride_oqd,
        q_out1,
    )
    tl.store(
        oq_base + head_offs[:, None] * stride_oqh + idx_odd[None, :] * stride_oqd,
        q_out2,
    )

    # ================= K (unique Hk key heads only) =================
    kh_offs = tl.arange(0, Hk)
    k_base = key_ptr + pid_t * stride_kt

    k1 = tl.load(k_base + kh_offs[:, None] * stride_kh + idx_even[None, :] * stride_kd)
    k2 = tl.load(k_base + kh_offs[:, None] * stride_kh + idx_odd[None, :] * stride_kd)

    k_out1 = (k1 * cos[None, :]) - (k2 * sin[None, :])
    k_out2 = (k1 * sin[None, :]) + (k2 * cos[None, :])

    ok_base = out_k_ptr + pid_t * stride_okt
    tl.store(
        ok_base + kh_offs[:, None] * stride_okh + idx_even[None, :] * stride_okd, k_out1
    )
    tl.store(
        ok_base + kh_offs[:, None] * stride_okh + idx_odd[None, :] * stride_okd, k_out2
    )

    # ================= pass-through (compile-time pruning) =================
    if D_HEAD > D_ROPE:
        dp = tl.arange(0, D_HEAD - D_ROPE)
        # Q pass-through
        q_pass = tl.load(
            q_base + head_offs[:, None] * stride_qh + (dp + D_ROPE)[None, :] * stride_qd
        )
        tl.store(
            oq_base
            + head_offs[:, None] * stride_oqh
            + (dp + D_ROPE)[None, :] * stride_oqd,
            q_pass,
        )
        # K pass-through
        k_pass = tl.load(
            k_base + kh_offs[:, None] * stride_kh + (dp + D_ROPE)[None, :] * stride_kd
        )
        tl.store(
            ok_base
            + kh_offs[:, None] * stride_okh
            + (dp + D_ROPE)[None, :] * stride_okd,
            k_pass,
        )


def fused_rope_qk_mqa(query, key, cos_sin, rotary_dim, is_neox_style):
    T, Hq, D = query.shape
    _, Hk, _ = key.shape

    out_q = torch.empty_like(query)
    out_k = torch.empty_like(key)

    grid = (T,)

    fused_rope_qk_mqa_kernel_opt[grid](
        query,
        key,
        cos_sin,
        out_q,
        out_k,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        cos_sin.stride(0),
        cos_sin.stride(1),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        Hq=Hq,
        Hk=Hk,
        D_HEAD=D,
        D_ROPE=rotary_dim,
        IS_NEOX_STYLE=is_neox_style,
    )

    return out_q, out_k
