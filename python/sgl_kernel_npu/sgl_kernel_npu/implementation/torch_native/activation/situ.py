from typing import Optional

import torch


def situ(
    hidden_states,
    group_list,
    group_list_type,
    *,
    need_quant,
    beta=4.0,
    linear_beta: Optional[float] = 25.0,
):
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    if hidden_states.ndim != 2 or hidden_states.shape[1] % 2:
        raise ValueError("SiTU input must have shape [tokens, 2 * intermediate]")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise ValueError("group_list must use int32 or int64")
    rows = int(
        group_list.as_strided((group_list.numel() + 1,), (1,))[-1]
        if group_list_type == 0
        else group_list.sum()
    )
    gate, up = hidden_states[:rows].float().chunk(2, -1)
    gate = beta * torch.tanh(gate / beta) * gate.sigmoid()
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    value = gate * up
    out = torch.empty(
        (hidden_states.shape[0], hidden_states.shape[1] // 2),
        dtype=torch.int8 if need_quant else hidden_states.dtype,
        device=hidden_states.device,
    )
    if not need_quant:
        out[:rows] = value.to(out.dtype)
        return out, None
    scale = torch.empty(
        hidden_states.shape[0], dtype=torch.float32, device=hidden_states.device
    )
    scale[:rows] = (value.abs().amax(-1) / 127).clamp_min(1e-30)
    out[:rows] = (
        torch.floor(value / scale[:rows, None] + 0.5).clamp(-128, 127).to(torch.int8)
    )
    return out, scale
