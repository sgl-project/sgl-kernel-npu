"""Independent FP32 PyTorch reference for SiTU and gate/up multiplication."""

from typing import Optional

import torch


def situ_and_mul(
    x: torch.Tensor,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: Optional[int] = None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
) -> torch.Tensor:
    """Assume valid inputs; leave rows beyond the routed prefix unspecified."""
    width = x.shape[-1]
    rows = x.numel() // width
    x_2d = x.reshape(rows, width)
    valid_rows = rows
    if group_list is not None:
        # Host synchronization is acceptable in this independent reference.
        total = group_list[-1] if group_list_type == 0 else group_list.sum()
        valid_rows = int(total.item())
    gate, up = x_2d[:valid_rows].float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    output = torch.empty((rows, width // 2), dtype=x.dtype, device=x.device)
    output[:valid_rows] = (gate * up).to(x.dtype)
    return output.reshape(*x.shape[:-1], width // 2)
