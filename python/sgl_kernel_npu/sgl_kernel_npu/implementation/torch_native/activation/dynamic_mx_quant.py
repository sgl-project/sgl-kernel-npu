"""Pure PyTorch reference for dynamic MXFP8 quantization.

Reproduces the vendor ``torch.ops.npu.npu_dynamic_mx_quant`` semantics with
``axis=-1`` and a fixed 32-element block using plain tensor arithmetic: each
group of 32 consecutive values along the last dimension shares one E8M0
(power-of-two) scale derived from the group's maximum exponent, and the scaled
values are saturated to +-448 before the E4M3FN cast. An all-zero group stores
scale byte ``0``; a group containing non-finite values stores scale byte
``255`` and a NaN payload.
"""

import torch

_MX_BLOCK = 32
_E4M3_MAX = 448.0
_E4M3_MAX_EXP = 8  # 448 = 1.75 * 2**8
_E8M0_BIAS = 127
_E8M0_NAN = 255


def _group_scales(blocks):
    """Return per-group E8M0 bytes and the finite-group mask.

    ``blocks`` is fp32 with shape ``[..., D // 32, 32]``. The maximum absolute
    value's floor(log2) determines the shared power-of-two scale, clamped so
    the largest scaled value fits the E4M3FN range.
    """
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
    scale_bytes = (exponent - float(_E4M3_MAX_EXP) + _E8M0_BIAS).clamp(min=0.0)
    scale_bytes = torch.where(nonzero, scale_bytes, torch.zeros_like(scale_bytes))
    scale_bytes = torch.where(
        finite, scale_bytes, torch.full_like(scale_bytes, _E8M0_NAN)
    )
    return scale_bytes.to(torch.uint8), finite


def dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn):
    """Reference implementation of per-32-element dynamic MX quantization.

    Returns an E4M3FN payload and per-32 scale bytes packed in pairs, with
    shape ``x.shape[:-1] + (ceil(D / 64), 2)`` and dtype ``torch.uint8``.
    """
    if dst_type != torch.float8_e4m3fn:
        raise ValueError(f"dst_type must be torch.float8_e4m3fn, got {dst_type}")
    import torch.nn.functional as F

    head_dim = x.shape[-1]
    padded = F.pad(x.to(torch.float32), (0, (-head_dim) % _MX_BLOCK))
    blocks = padded.reshape(*x.shape[:-1], -1, _MX_BLOCK)
    scale_bytes, finite = _group_scales(blocks)
    scale = torch.pow(2.0, scale_bytes.to(torch.float32) - float(_E8M0_BIAS)).unsqueeze(
        -1
    )
    scaled = (blocks / scale).clamp(-_E4M3_MAX, _E4M3_MAX)
    nan = torch.full_like(scaled, float("nan"))
    scaled = torch.where(finite.unsqueeze(-1), scaled, nan)
    payload = scaled.flatten(-2)[..., :head_dim].to(torch.float8_e4m3fn)
    scales = F.pad(scale_bytes, (0, scale_bytes.shape[-1] % 2))
    scales = scales.reshape(*x.shape[:-1], -1, 2)
    return payload, scales
