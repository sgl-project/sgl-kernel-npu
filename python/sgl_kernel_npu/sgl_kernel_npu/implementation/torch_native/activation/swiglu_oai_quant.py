"""Independent PyTorch reference for split-layout SwiGLU-OAI quantization."""

import torch


def swiglu_oai_quant(
    x, alpha, limit, need_quant=True, group_list=None, group_list_type=None
):
    """Compute valid rows in FP32; assume valid input metadata and group values."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    rows, cols = flat.shape
    valid_rows = rows
    if group_list is not None:
        valid_rows = int(
            (group_list[-1] if group_list_type == 0 else group_list.sum()).item()
        )
    gate, up = flat[:valid_rows].float().chunk(2, dim=-1)
    gate = gate.clamp(max=float(limit))
    up = up.clamp(min=-float(limit), max=float(limit))
    values = gate * torch.sigmoid(gate * float(alpha)) * (up + 1.0)
    out = torch.empty(
        (rows, cols // 2), dtype=torch.int8 if need_quant else x.dtype, device=x.device
    )
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if need_quant:
        row_scale = values.abs().amax(dim=-1) / 127.0
        divisor = torch.where(row_scale == 0, 1.0, row_scale)
        # Preserve the kernel's round-trip through input precision before the
        # saturating INT8 cast (round toward zero, not round-to-nearest).
        normalized = (values / divisor.unsqueeze(-1)).to(x.dtype).float()
        out[:valid_rows] = normalized.trunc().clamp(-128, 127).to(torch.int8)
        scale[:valid_rows] = row_scale
    else:
        out[:valid_rows] = values.to(x.dtype)
    return out.reshape(*shape[:-1], cols // 2), scale
