import torch
import triton
import triton.language as tl


@triton.jit
def _fused_conv1d_decode_packed_gating_kernel(
    QKVZ,  # [B, QKVZ_DIM]
    BA,  # [B, 2*NV] (b in [0:NV], a in [NV:2*NV])
    W,  # [4, QKV_DIM]
    STATE,  # [num_slots, 3, QKV_DIM]
    CACHE_IDX,  # [B] int32/int64
    A_LOG,  # [NV] fp32
    DT_BIAS,  # [NV] bf16/fp32
    MIXED_OUT,  # [B, QKV_DIM]
    G_OUT,  # [B, NV] fp32
    BETA_OUT,  # [B, NV] bf16/fp16
    B,
    QKVZ_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    NV: tl.constexpr,
    NV_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n_chunks = QKV_DIM // BLOCK_D
    pid = tl.program_id(0)
    b = pid // n_chunks
    c = pid % n_chunks
    b64 = b.to(tl.int64)
    slot = tl.load(CACHE_IDX + b).to(tl.int64)
    if c == 0:
        offs_nv = tl.arange(0, NV_PAD)
        mask_nv = offs_nv < NV
        b_val = tl.load(BA + b64 * (2 * NV) + offs_nv, mask=mask_nv, other=0.0).to(
            tl.float32
        )
        a_val = tl.load(BA + b64 * (2 * NV) + NV + offs_nv, mask=mask_nv, other=0.0).to(
            tl.float32
        )
        a_log = tl.load(A_LOG + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)
        dt_b = tl.load(DT_BIAS + offs_nv, mask=mask_nv, other=0.0).to(tl.float32)
        sp_in = a_val + dt_b
        sp = tl.where(sp_in > 20.0, sp_in, tl.log(1.0 + tl.exp(sp_in)))
        g_val = -tl.exp(a_log) * sp
        # AscendC recurrent_gated_delta_rule expects pre-sigmoid beta logits.
        tl.store(G_OUT + b64 * NV + offs_nv, g_val, mask=mask_nv)
        tl.store(
            BETA_OUT + b64 * NV + offs_nv,
            b_val.to(BETA_OUT.dtype.element_ty),
            mask=mask_nv,
        )

    offs = c * BLOCK_D + tl.arange(0, BLOCK_D)
    x0 = tl.load(QKVZ + b64 * QKVZ_DIM + offs)
    if slot >= 0:
        s_base_slot = slot * (3 * QKV_DIM)
        s0 = tl.load(STATE + s_base_slot + offs)
        s1 = tl.load(STATE + s_base_slot + QKV_DIM + offs)
        s2 = tl.load(STATE + s_base_slot + 2 * QKV_DIM + offs)
        tl.store(STATE + s_base_slot + offs, s1)
        tl.store(STATE + s_base_slot + QKV_DIM + offs, s2)
        tl.store(STATE + s_base_slot + 2 * QKV_DIM + offs, x0)
        w0 = tl.load(W + offs).to(tl.float32)
        w1 = tl.load(W + QKV_DIM + offs).to(tl.float32)
        w2 = tl.load(W + 2 * QKV_DIM + offs).to(tl.float32)
        w3 = tl.load(W + 3 * QKV_DIM + offs).to(tl.float32)
        acc = (
            s0.to(tl.float32) * w0
            + s1.to(tl.float32) * w1
            + s2.to(tl.float32) * w2
            + x0.to(tl.float32) * w3
        )
        out = acc * tl.sigmoid(acc)
        tl.store(MIXED_OUT + b64 * QKV_DIM + offs, out.to(MIXED_OUT.dtype.element_ty))
    else:
        tl.store(MIXED_OUT + b64 * QKV_DIM + offs, x0)


def fused_conv1d_decode_gating(
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    conv_weights_t: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_v_heads: int,
):
    """Fused decode depthwise causal conv1d (kernel=4, SiLU) + in-place state shift + GDN gating."""
    if projected_states_qkvz.ndim != 2 or not projected_states_qkvz.is_contiguous():
        raise ValueError("projected_states_qkvz must be contiguous 2D [B, QKVZ_DIM]")
    if projected_states_ba.ndim != 2 or not projected_states_ba.is_contiguous():
        raise ValueError("projected_states_ba must be contiguous 2D [B, 2*NV]")
    if (
        conv_weights_t.ndim != 2
        or conv_weights_t.shape[0] != 4
        or not conv_weights_t.is_contiguous()
    ):
        raise ValueError("conv_weights_t must be contiguous 2D [4, QKV_DIM]")
    if (
        conv_state.ndim != 3
        or conv_state.shape[1] != 3
        or not conv_state.is_contiguous()
    ):
        raise ValueError("conv_state must be contiguous 3D [num_slots, 3, QKV_DIM]")
    if cache_indices.ndim != 1 or not cache_indices.is_contiguous():
        raise ValueError("cache_indices must be contiguous 1D [B]")

    bs = int(projected_states_qkvz.shape[0])
    qkvz_dim = int(projected_states_qkvz.shape[1])
    qkv_dim = int(conv_weights_t.shape[1])
    if qkv_dim % 512 != 0:
        raise ValueError(f"qkv_dim must be a multiple of 512, got {qkv_dim}")
    if conv_state.shape[2] != qkv_dim:
        raise ValueError(
            f"conv_state.shape[2] ({conv_state.shape[2]}) != qkv_dim ({qkv_dim})"
        )
    if projected_states_ba.shape != (bs, 2 * num_v_heads):
        raise ValueError(
            f"projected_states_ba shape {tuple(projected_states_ba.shape)} != {(bs, 2 * num_v_heads)}"
        )

    dev = projected_states_qkvz.device
    mixed_out = torch.empty(
        (bs, qkv_dim), dtype=projected_states_qkvz.dtype, device=dev
    )
    g_out = torch.empty((bs, num_v_heads), dtype=torch.float32, device=dev)
    beta_out = torch.empty(
        (bs, num_v_heads), dtype=projected_states_qkvz.dtype, device=dev
    )
    if bs == 0:
        return mixed_out, g_out, beta_out

    if bs >= 2 and qkv_dim % 2048 == 0:
        block_d = 2048
    elif qkv_dim % 1024 == 0:
        block_d = 1024
    else:
        block_d = 512
    nv_pad = triton.next_power_of_2(num_v_heads)
    _fused_conv1d_decode_packed_gating_kernel[(bs * (qkv_dim // block_d),)](
        projected_states_qkvz,
        projected_states_ba,
        conv_weights_t,
        conv_state,
        cache_indices,
        A_log,
        dt_bias,
        mixed_out,
        g_out,
        beta_out,
        bs,
        qkvz_dim,
        qkv_dim,
        num_v_heads,
        nv_pad,
        block_d,
    )
    return mixed_out, g_out, beta_out
