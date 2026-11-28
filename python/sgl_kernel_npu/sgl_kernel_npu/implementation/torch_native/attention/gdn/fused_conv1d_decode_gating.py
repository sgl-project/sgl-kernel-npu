import torch
import torch.nn.functional as F


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
    """Reference PyTorch implementation of decode causal conv1d (k=4, SiLU) + gating."""
    bs = int(projected_states_qkvz.shape[0])
    qkv_dim = int(conv_weights_t.shape[1])
    dev = projected_states_qkvz.device
    out_dtype = projected_states_qkvz.dtype

    mixed_out = torch.empty((bs, qkv_dim), dtype=out_dtype, device=dev)
    b_raw = projected_states_ba[:, :num_v_heads]
    a_raw = projected_states_ba[:, num_v_heads:].float()
    g_out = -torch.exp(A_log.float()) * F.softplus(a_raw + dt_bias.float())
    beta_out = b_raw.to(out_dtype).contiguous()

    w_f = conv_weights_t.float()
    for bi in range(bs):
        slot = int(cache_indices[bi].item())
        x0 = projected_states_qkvz[bi, :qkv_dim]
        if slot >= 0:
            s_old = conv_state[slot].clone()
            acc = (
                s_old[0].float() * w_f[0]
                + s_old[1].float() * w_f[1]
                + s_old[2].float() * w_f[2]
                + x0.float() * w_f[3]
            )
            mixed_out[bi] = (acc * torch.sigmoid(acc)).to(out_dtype)
            conv_state[slot, 0] = s_old[1]
            conv_state[slot, 1] = s_old[2]
            conv_state[slot, 2] = x0
        else:
            mixed_out[bi] = x0

    return mixed_out, g_out.to(torch.float32), beta_out
