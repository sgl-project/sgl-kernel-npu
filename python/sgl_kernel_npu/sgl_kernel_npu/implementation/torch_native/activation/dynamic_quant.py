"""Pure PyTorch reference for per-row dynamic INT8 quantization.

Reproduces the vendor ``torch_npu.npu_dynamic_quant`` semantics with plain
tensor arithmetic: each row (last dimension) shares one symmetric scale
``amax / 127``; the scaled values are rounded and saturated to +-127 before the
INT8 cast. An all-zero row keeps scale ``0`` and a zero payload.
"""

import torch

_INT8_MAX = 127


def dynamic_quant(x, dst_type=torch.int8):
    """Reference implementation of per-row dynamic INT8 quantization.

    ``x`` is a contiguous bf16/fp16/fp32 tensor with a positive last dimension.
    Returns ``(payload, scale)``: the payload has the same shape as ``x`` in
    ``torch.int8``, and the scale has shape ``x.shape[:-1]`` in
    ``torch.float32`` (one ``amax / 127`` scale per row).
    """
    if dst_type != torch.int8:
        raise ValueError(f"dst_type must be torch.int8, got {dst_type}")
    rows = x.to(torch.float32).reshape(-1, x.shape[-1])
    scale = rows.abs().amax(dim=-1) / float(_INT8_MAX)
    # All-zero rows keep scale 0; divide by 1 so the payload stays 0.
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    payload = (
        torch.round(rows / divisor.unsqueeze(-1))
        .clamp(-_INT8_MAX, _INT8_MAX)
        .to(torch.int8)
    )
    return payload.reshape(x.shape), scale.reshape(x.shape[:-1])
