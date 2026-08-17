from typing import Optional

import torch


def laser_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
    alibi_mask: Optional[torch.Tensor] = None,
    drop_mask: Optional[torch.Tensor] = None,
    scale_value: float = 1.0,
    head_num: int = 2,
    input_layout: str = "BNSD",
    keep_prob: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 1,
    is_highPrecision: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Laser Attention on FP16 BNSD inputs.

    The current kernel requires head dimension 128 and sequence lengths that
    are multiples of 128. Both returned tensors use FP32.
    """
    return torch.ops.npu.laser_attn(
        query,
        key,
        value,
        atten_mask,
        alibi_mask,
        drop_mask,
        scale_value,
        head_num,
        input_layout,
        keep_prob,
        pre_tokens,
        next_tokens,
        is_highPrecision,
    )


__all__ = ["laser_attn"]
