import torch


def _select_mrope_cos_sin(
    cos_sin: torch.Tensor,
    mrope_section: list[int],
    is_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_dim = cos_sin.shape[-1]
    half_rope_dim = rope_dim // 2
    offsets = torch.arange(half_rope_dim, device=cos_sin.device)

    if is_interleaved:
        h_mask = (offsets % 3 == 1) & (offsets <= 3 * mrope_section[1])
        w_mask = (offsets % 3 == 2) & (offsets <= 3 * mrope_section[2])
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section[0]
        h_end = t_end + mrope_section[1]
        w_end = h_end + mrope_section[2]
        t_mask = offsets < t_end
        h_mask = (offsets >= t_end) & (offsets < h_end)
        w_mask = (offsets >= h_end) & (offsets < w_end)

    cos = torch.where(
        t_mask,
        cos_sin[0, :, :half_rope_dim],
        torch.where(
            h_mask,
            cos_sin[1, :, :half_rope_dim],
            cos_sin[2, :, :half_rope_dim],
        ),
    )
    sin = torch.where(
        t_mask,
        cos_sin[0, :, half_rope_dim:],
        torch.where(
            h_mask,
            cos_sin[1, :, half_rope_dim:],
            cos_sin[2, :, half_rope_dim:],
        ),
    )
    return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    x = x * weight.float()
    if bias is not None:
        x = x + bias.float()
    return x


def _apply_mrope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    rotary = x[..., :rope_dim]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    rotary = rotary * cos[:, None, :] + rotated * sin[:, None, :]
    return torch.cat((rotary, x[..., rope_dim:]), dim=-1)


def split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rope_dim is None:
        rope_dim = head_size
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    if has_gate:
        q_gate, k, v = qkv.split((q_size * 2, kv_size, kv_size), dim=-1)
        q, gate = q_gate.view(num_tokens, num_q_heads, head_size * 2).chunk(2, dim=-1)
        gate = gate.reshape(num_tokens, q_size)
    else:
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(num_tokens, num_q_heads, head_size)
        gate = qkv.new_empty((num_tokens, 0))

    k = k.view(num_tokens, num_kv_heads, head_size)
    q = _rms_norm(q, q_weight, eps, q_bias)
    k = _rms_norm(k, k_weight, eps, k_bias)
    cos, sin = _select_mrope_cos_sin(cos_sin, mrope_section, is_interleaved)
    q = _apply_mrope(q, cos.float(), sin.float(), rope_dim)
    k = _apply_mrope(k, cos.float(), sin.float(), rope_dim)
    return (
        q.flatten(1).to(qkv.dtype),
        k.flatten(1).to(qkv.dtype),
        v.clone(),
        gate.clone(),
    )
