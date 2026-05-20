import torch
import torch.nn.functional as F


def swiglu_silu_clamp_mul_native(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """Out-variant of swiglustep activation.

    Writes into `out`:
      silu(x[:d]).clamp(max=limit) * x[d:].clamp(-limit, limit)
    """
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    out = gate * up
    return out


def swiglu_silu_clamp_mul(hidden_states, limit: float = 7.0) -> torch.Tensor:
    return swiglu_silu_clamp_mul_native(
        hidden_states,
        limit,
    )
