import torch
import triton
import triton.language as tl


@triton.jit
def fused_split_qk_norm_kernel(
    fused_ptr,  # [B, total_hidden]
    q_lora_ptr,  # [B, q_lora]
    k_nope_ptr,  # [B, kv_lora]
    k_pe_ptr,  # [B, qk_rope]
    q_rms_w_ptr,
    q_rms_b_ptr,
    k_rms_w_ptr,
    k_rms_b_ptr,
    total_hidden: tl.constexpr,
    q_lora_rank: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_dim: tl.constexpr,
    eps: tl.constexpr,
    Q_HAS_BIAS: tl.constexpr,
    K_HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)

    base = pid * total_hidden

    # =====================================================
    # Q LORA  (RMSNorm)
    # =====================================================
    q_offs = tl.arange(0, q_lora_rank)
    q = tl.load(
        fused_ptr + base + q_offs,
        mask=q_offs < q_lora_rank,
        other=0.0,
    ).to(tl.float32)

    q_var = tl.sum(q * q, axis=0) / q_lora_rank
    q_rstd = tl.rsqrt(q_var + eps)

    qw = tl.load(q_rms_w_ptr + q_offs, mask=q_offs < q_lora_rank)
    q = q * q_rstd * qw

    if Q_HAS_BIAS:
        qb = tl.load(q_rms_b_ptr + q_offs, mask=q_offs < q_lora_rank)
        q += qb

    tl.store(q_lora_ptr + pid * q_lora_rank + q_offs, q, mask=q_offs < q_lora_rank)

    # =====================================================
    # K NOPE  (RMSNorm)
    # =====================================================
    k_base = base + q_lora_rank
    k_offs = tl.arange(0, kv_lora_rank)

    k = tl.load(
        fused_ptr + k_base + k_offs,
        mask=k_offs < kv_lora_rank,
        other=0.0,
    ).to(tl.float32)

    k_var = tl.sum(k * k, axis=0) / kv_lora_rank
    k_rstd = tl.rsqrt(k_var + eps)

    kw = tl.load(k_rms_w_ptr + k_offs, mask=k_offs < kv_lora_rank)
    k = k * k_rstd * kw

    if K_HAS_BIAS:
        kb = tl.load(k_rms_b_ptr + k_offs, mask=k_offs < kv_lora_rank)
        k += kb

    tl.store(k_nope_ptr + pid * kv_lora_rank + k_offs, k, mask=k_offs < kv_lora_rank)

    # =====================================================
    # K PE  (no norm, direct copy)
    # =====================================================
    pe_offs = tl.arange(0, qk_rope_dim)
    pe_base = k_base + kv_lora_rank

    k_pe = tl.load(
        fused_ptr + pe_base + pe_offs,
        mask=pe_offs < qk_rope_dim,
    )

    tl.store(
        k_pe_ptr + pid * qk_rope_dim + pe_offs,
        k_pe,
        mask=pe_offs < qk_rope_dim,
    )


def fused_split_qk_norm(
    fused_qkv_a_proj_out,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_dim,
    eps=1e-6,
):
    B, total_hidden = fused_qkv_a_proj_out.shape
    device = fused_qkv_a_proj_out.device
    dtype = fused_qkv_a_proj_out.dtype

    q_lora = torch.empty((B, q_lora_rank), device=device, dtype=dtype)
    k_nope = torch.empty((B, kv_lora_rank), device=device, dtype=dtype)
    k_pe = torch.empty((B, qk_rope_dim), device=device, dtype=dtype)

    fused_split_qk_norm_kernel[(B,)](
        fused_qkv_a_proj_out,
        q_lora,
        k_nope,
        k_pe,
        q_a_layernorm.weight,
        q_a_layernorm.bias,
        kv_a_layernorm.weight,
        kv_a_layernorm.bias,
        total_hidden,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_dim,
        eps,
        q_a_layernorm.bias is not None,
        kv_a_layernorm.bias is not None,
    )

    # Restore original shape by adding a sequence dimension
    k_nope = k_nope.unsqueeze(1)
    k_pe = k_pe.unsqueeze(1)

    return q_lora, k_nope, k_pe
