"""Shared MXFP8 quantization for KPool pooled-key cache writes.

Reuses the E8M0 group-scale math from
``implementation/torch_native/activation/situ_mxfp8_quant.py``: one power-of-two
scale per 32 consecutive channels, an E4M3FN payload, and raw E8M0 scale bytes
packed as ``(D/64, 2)`` per cache row.
"""

import torch

MX_BLOCK = 32
E4M3_MAX = 448.0
E4M3_MAX_EXP = 8  # 448 = 1.75 * 2**8
E8M0_BIAS = 127
E8M0_NAN = 255


def mx_quantize_row(pooled):
    """Quantize one FP32 pooled row ``[D]`` into E4M3FN payload + E8M0 bytes.

    Returns ``(payload, scale_bytes)`` where ``payload`` is ``float8_e4m3fn``
    with shape ``[D]`` and ``scale_bytes`` is ``uint8`` shaped ``(D/64, 2)``.
    """
    head_dim = pooled.shape[0]
    blocks = pooled.to(torch.float32).reshape(head_dim // MX_BLOCK, MX_BLOCK)
    group_max = blocks.abs().amax(dim=-1)
    finite = torch.isfinite(group_max)
    nonzero = group_max > 0
    magnitude = torch.where(nonzero & finite, group_max, torch.ones_like(group_max))
    exponent = torch.floor(torch.log2(magnitude))
    # floor(log2) may land on the wrong side of an exact power of two.
    exponent = torch.where(
        torch.pow(2.0, exponent) > magnitude, exponent - 1.0, exponent
    )
    exponent = torch.where(
        torch.pow(2.0, exponent + 1.0) <= magnitude, exponent + 1.0, exponent
    )
    scale_bytes = (exponent - float(E4M3_MAX_EXP) + E8M0_BIAS).clamp(min=0.0)
    scale_bytes = torch.where(nonzero, scale_bytes, torch.zeros_like(scale_bytes))
    scale_bytes = torch.where(
        finite, scale_bytes, torch.full_like(scale_bytes, E8M0_NAN)
    )
    scale_bytes = scale_bytes.to(torch.uint8)
    scale = torch.pow(2.0, scale_bytes.to(torch.float32) - float(E8M0_BIAS)).unsqueeze(
        -1
    )
    scaled = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX)
    scaled = torch.where(
        finite.unsqueeze(-1), scaled, torch.full_like(scaled, float("nan"))
    )
    payload = scaled.reshape(head_dim).to(torch.float8_e4m3fn)
    return payload, scale_bytes.reshape(head_dim // 64, 2)
