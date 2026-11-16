import torch
import torch.nn.functional as F


def swiglu_quant(
    x, group_list, group_list_type, need_quant=True, do_limit=False, limit=7.0
):
    if group_list_type not in [0, 1]:
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"group_list dtype must be torch.int32 or torch.int64, but got {group_list.dtype}"
        )
    rows = int(
        group_list.as_strided((group_list.numel() + 1,), (1,))[-1]
        if group_list_type == 0
        else group_list.sum()
    )
    gate, up = x[:rows].float().chunk(2, dim=-1)
    if do_limit:
        value = F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
    else:
        value = gate * torch.sigmoid(gate) * up
    out = torch.empty(
        (x.shape[0], x.shape[1] // 2),
        dtype=torch.int8 if need_quant else x.dtype,
        device=x.device,
    )
    scale = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    if need_quant:
        scale[:rows] = value.abs().amax(-1) / 127
        out[:rows] = (
            torch.floor(value / scale[:rows, None] + 0.5)
            .clamp(-127, 127)
            .to(torch.int8)
        )
    else:
        out[:rows] = value.to(x.dtype)
    return out, scale
