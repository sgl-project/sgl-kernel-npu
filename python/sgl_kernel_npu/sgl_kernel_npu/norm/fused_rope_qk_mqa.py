import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_qk_mqa_kernel(
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
    Hk: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_ROPE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    # MQA: key head broadcast
    kh = pid_h % Hk

    # -------- rotary indices
    d = tl.arange(0, D_ROPE // 2)
    if IS_NEOX_STYLE:
        idx_even = d
        idx_odd = d + D_ROPE // 2
    else:
        idx_even = d * 2
        idx_odd = d * 2 + 1

    # cos / sin
    cos = tl.load(cos_sin_ptr + pid_t * stride_ct + d * stride_cd)
    sin = tl.load(cos_sin_ptr + pid_t * stride_ct + (d + D_ROPE // 2) * stride_cd)

    # ================= Q =================
    q_base = query_ptr + pid_t * stride_qt + pid_h * stride_qh

    q1 = tl.load(q_base + idx_even * stride_qd)
    q2 = tl.load(q_base + idx_odd * stride_qd)

    q_out1 = (q1 * cos) - (q2 * sin)
    q_out2 = (q1 * sin) + (q2 * cos)

    oq_base = out_q_ptr + pid_t * stride_oqt + pid_h * stride_oqh
    tl.store(oq_base + idx_even * stride_oqd, q_out1)
    tl.store(oq_base + idx_odd * stride_oqd, q_out2)

    # ================= K =================
    k_base = key_ptr + pid_t * stride_kt + kh * stride_kh
    k1 = tl.load(k_base + idx_even * stride_kd)
    k2 = tl.load(k_base + idx_odd * stride_kd)

    k_out1 = (k1 * cos) - (k2 * sin)
    k_out2 = (k1 * sin) + (k2 * cos)

    ok_base = out_k_ptr + pid_t * stride_okt + kh * stride_okh
    tl.store(ok_base + idx_even * stride_okd, k_out1)
    tl.store(ok_base + idx_odd * stride_okd, k_out2)

    # ================= pass-through (compile-time pruning) =================
    if D_HEAD > D_ROPE:
        dp = tl.arange(0, D_HEAD - D_ROPE)
        tl.store(
            oq_base + (dp + D_ROPE) * stride_oqd,
            tl.load(q_base + (dp + D_ROPE) * stride_qd),
        )
        tl.store(
            ok_base + (dp + D_ROPE) * stride_okd,
            tl.load(k_base + (dp + D_ROPE) * stride_kd),
        )


def fused_rope_qk_mqa(
    query,
    key,
    cos_sin,
    rotary_dim,
    is_neox_style,  # [T, Hq, D]  # [T, Hk, D]  # [T, rotary_dim]
):
    T, Hq, D = query.shape
    _, Hk, _ = key.shape

    out_q = torch.empty_like(query)
    out_k = torch.empty_like(key)

    grid = (T, Hq)

    fused_rope_qk_mqa_kernel[grid](
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
        Hk=Hk,
        D_HEAD=D,
        D_ROPE=rotary_dim,
        IS_NEOX_STYLE=is_neox_style,
    )

    return out_q, out_k
