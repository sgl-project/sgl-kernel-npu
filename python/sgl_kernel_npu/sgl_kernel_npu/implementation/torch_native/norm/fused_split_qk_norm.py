"""Split MLA latent projections and normalize Q/K without Triton."""

from .norm_utils import rms_norm, validate_split_qk_norm


def fused_split_qk_norm(
    fused_qkv_a_proj_out,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_dim,
    eps=1e-6,
):
    validate_split_qk_norm(
        fused_qkv_a_proj_out,
        q_a_layernorm,
        kv_a_layernorm,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_dim,
        eps,
    )
    q, k, pe = fused_qkv_a_proj_out.split(
        (q_lora_rank, kv_lora_rank, qk_rope_dim), dim=-1
    )
    q = rms_norm(q, q_a_layernorm.weight, eps, getattr(q_a_layernorm, "bias", None))
    k = rms_norm(k, kv_a_layernorm.weight, eps, getattr(kv_a_layernorm, "bias", None))
    dtype = fused_qkv_a_proj_out.dtype
    return q.to(dtype), k.to(dtype).unsqueeze(1), pe.unsqueeze(1).clone().contiguous()
