import torch
import triton
import triton.language as tl


@triton.jit
def partial_rope_qk_inplace_kernel(
    query_ptr,
    key_ptr,
    cos_sin_ptr,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_ct,
    stride_cd,
    groups: tl.constexpr,
    D_ROPE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    t_id = tl.program_id(0)
    hk_id = tl.program_id(1)

    d = tl.arange(0, D_ROPE // 2)
    if IS_NEOX_STYLE:
        idx_even = d
        idx_odd = d + D_ROPE // 2
    else:
        idx_even = d * 2
        idx_odd = d * 2 + 1

    # cos / sin
    cos = tl.load(cos_sin_ptr + t_id * stride_ct + d * stride_cd)  # (D_ROPE // 2,)
    sin = tl.load(cos_sin_ptr + t_id * stride_ct + (d + D_ROPE // 2) * stride_cd)  # (D_ROPE // 2,)

    # ================= Q =================
    for g_id in range(groups):
        hq_id = hk_id + g_id
        q_base = query_ptr + t_id * stride_qt + hq_id * stride_qh
        q1 = tl.load(q_base + idx_even * stride_qd)
        q2 = tl.load(q_base + idx_odd * stride_qd)
        q_out1 = (q1 * cos) - (q2 * sin)
        q_out2 = (q1 * sin) + (q2 * cos)
        tl.store(q_base + idx_even * stride_qd, q_out1)
        tl.store(q_base + idx_odd * stride_qd, q_out2)

    # ================= K =================
    k_base = key_ptr + t_id * stride_kt + hk_id * stride_kh
    k1 = tl.load(k_base + idx_even * stride_kd)
    k2 = tl.load(k_base + idx_odd * stride_kd)

    k_out1 = (k1 * cos) - (k2 * sin)
    k_out2 = (k1 * sin) + (k2 * cos)

    tl.store(k_base + idx_even * stride_kd, k_out1)
    tl.store(k_base + idx_odd * stride_kd, k_out2)


def partial_rope_qk_inplace(
    query,          # [T, Hq, D]
    key,            # [T, Hk, D]
    cos_sin,        # [T, rotary_dim]
    rotary_dim,
    is_neox_style=False,
):
    T, Hq, D = query.shape
    _, Hk, _ = key.shape
    assert Hq % Hk == 0

    grid = (T, Hk)

    partial_rope_qk_inplace_kernel[grid](
        query,
        key,
        cos_sin,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        cos_sin.stride(0),
        cos_sin.stride(1),
        groups=Hq // Hk,
        D_ROPE=rotary_dim,
        IS_NEOX_STYLE=is_neox_style,
    )

    return query, key
