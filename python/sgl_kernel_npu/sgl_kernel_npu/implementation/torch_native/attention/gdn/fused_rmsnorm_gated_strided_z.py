import torch
import torch.nn.functional as F


def fused_rmsnorm_gated_strided_z(
    core_attn_out: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    norm_weight: torch.Tensor,
    z_offset: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference PyTorch implementation of RMSNorm(core_attn_out) * SiLU(z)."""
    T, num_v_heads, head_v_dim = core_attn_out.shape
    v_dim = num_v_heads * head_v_dim
    x_f = core_attn_out.float()
    rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    z_f = (
        projected_states_qkvz[:, z_offset : z_offset + v_dim]
        .view(T, num_v_heads, head_v_dim)
        .float()
    )
    y = (x_f * rstd) * norm_weight.float().view(1, 1, head_v_dim) * F.silu(z_f)
    return y.to(core_attn_out.dtype).reshape(T, v_dim).contiguous()
