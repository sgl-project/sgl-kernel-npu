# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel_npu.fla.utils import input_guard


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_npu_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BHV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k = 0
    i_v, i_n, i_nhv = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )  # 0 1
        all = T  # 1
        T = eos - bos  # 1
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)  # 0 ~ 127
    o_v = i_v * BV + tl.arange(0, BV)  # 0 ~ 15 * 8 + 0 ~ 7
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    for i_bhv in range(BHV):
        i_hv = i_nhv * BHV + i_bhv
        i_h = i_hv // (HV // H)
        p_q = q + (bos * H + i_h) * K + o_k  # q + 0 ~ 127
        p_k = k + (bos * H + i_h) * K + o_k  # k + 0 ~ 127
        p_v = v + (bos * HV + i_hv) * V + o_v  # v + i_nh * 128 + 0 ~ 15 * 8 + 0 ~ 7
        p_b = b + bos * HV + i_hv
        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

        # Gating computation pointers
        p_A_log = A_log + i_hv
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

        for i in range(T):
            # Load inputs
            b_q = tl.load(p_q + i * H * K, mask=mask_k).to(tl.float32)  # 128 * float32
            b_k = tl.load(p_k + i * H * K, mask=mask_k).to(tl.float32)  # 128 * float32
            b_v = tl.load(p_v + i * HV * V, mask=mask_v).to(tl.float32)  # 64 * float32
            b_b = tl.load(p_b + i * HV).to(tl.float32)

            # Compute sigmoid gating
            # Load gating parameters
            b_A_log = tl.load(p_A_log).to(tl.float32)
            b_a = tl.load(p_a + i * HV).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

            b_h = tl.zeros([BK, BV], dtype=tl.float32)
            if USE_INITIAL_STATE:
                idx = tl.load(h0_indices + i_n)  #
                if idx >= 0:
                    p_h0 = (
                        h0_source
                        + idx * HV * K * V
                        + i_hv * K * V
                        + o_k[:, None] * V
                        + o_v[None, :]
                    )  # 128 * 64 * int32
                    b_h = tl.load(p_h0, mask=mask_h).to(tl.float32)

            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a + b_dt_bias
            beta_x = softplus_beta * x
            # Apply softplus with numerical stability
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))

            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

            b_q = b_q * scale

            # Apply gating to hidden state: h *= exp(g)
            b_h *= tl.exp(b_g)

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= tl.sum(b_h * b_k[:, None], 0)

            # Apply beta gating: v *= beta
            b_v *= b_beta

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :]

            # Compute output: o = sum(h * q, dim=0)
            b_o = tl.sum(b_h * b_q[:, None], 0)

            if USE_INITIAL_STATE:
                idx = tl.load(h0_indices + i_n)
                if idx >= 0:
                    p_h0 = (
                        h0_source
                        + idx * HV * K * V
                        + i_hv * K * V
                        + o_k[:, None] * V
                        + o_v[None, :]
                    )  # 128 * 64 * int32
                    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

            tl.store(p_o + i * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)


@input_guard
def fused_sigmoid_gating_delta_rule_update_npu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    BHV = 2
    NHV = HV // BHV
    grid = (NV, N, NHV)

    fused_sigmoid_gating_delta_rule_update_npu_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BHV=BHV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
        multibuffer=False,
    )
    o = o.squeeze(0)
    return o
