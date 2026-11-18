"""Pure PyTorch reference for fused SiTU + MXFP8 quantization.

The Ascend C operator evaluates SiTU in fp32, rounds it to BF16, and then uses
the vendor micro-scaling routine from ``dynamic_mx_quant.h``. This module
reproduces that path with plain tensor arithmetic:

* SiTU uses ``tanh(x) = 2 * sigmoid(2x) - 1`` and applies the same factor order
  as the Kernel before the BF16 round trip.
* MXFP8 shares one E8M0 (power-of-two) scale per 32 consecutive BF16 values.
  The exponent is the maximum BF16 exponent of the group, clamped so the
  largest scaled value fits the E4M3FN range, and 255 marks non-finite groups.
"""

import torch

_GATE_COLS = 3072
_MX_BLOCK = 32
_E4M3_MAX = 448.0
_E4M3_MAX_EXP = 8  # 448 = 1.75 * 2**8
_E8M0_BIAS = 127
_E8M0_NAN = 255
_SCALE_COLS = _GATE_COLS // _MX_BLOCK


def _valid_rows(group_list, group_list_type):
    """Rows selected by either the cumulative or the per-expert count layout."""
    if group_list_type == 0:
        value = int(group_list[-1].item())
    else:
        value = int(group_list.sum().item())
    return max(value, 0)


def _situ(x, beta, linear_beta):
    """fp32 SiTU rounded to BF16, matching the unfused path's output dtype."""
    gate = x[:, :_GATE_COLS].to(torch.float32)
    up = x[:, _GATE_COLS:].to(torch.float32)
    gate_value = (2.0 * beta) * torch.sigmoid(gate * (2.0 / beta)) - beta
    up_value = (2.0 * linear_beta) * torch.sigmoid(
        up * (2.0 / linear_beta)
    ) - linear_beta
    return (torch.sigmoid(gate) * gate_value * up_value).to(torch.bfloat16)


def _group_scales(value):
    """Return per-group E8M0 bytes and the finite-group mask.

    ``value`` is BF16. Only its exponent matters, so the maximum absolute
    value's floor(log2) determines the shared power-of-two scale.
    """
    rows, cols = value.shape
    blocks = value.to(torch.float32).reshape(rows, cols // _MX_BLOCK, _MX_BLOCK)
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


def _quantize(value):
    """Quantize BF16 values into E4M3FN payload plus E8M0 scale bytes."""
    rows, cols = value.shape
    blocks = value.to(torch.float32).reshape(rows, cols // _MX_BLOCK, _MX_BLOCK)
    scale_bytes, finite = _group_scales(value)
    scale = torch.pow(2.0, scale_bytes.to(torch.float32) - float(_E8M0_BIAS)).unsqueeze(
        -1
    )
    scaled = (blocks / scale).clamp(-_E4M3_MAX, _E4M3_MAX)
    nan = torch.full_like(scaled, float("nan"))
    scaled = torch.where(finite.unsqueeze(-1), scaled, nan)
    return scaled.reshape(rows, cols).to(torch.float8_e4m3fn), scale_bytes


def situ_mxfp8_quant(
    hidden_states,
    group_list,
    group_list_type=1,
    beta=4.0,
    linear_beta=25.0,
):
    """Reference implementation of the fused SiTU + MXFP8 quantization."""
    capacity = hidden_states.shape[0]
    valid_rows = min(_valid_rows(group_list, group_list_type), capacity)
    payload = torch.empty(
        (capacity, _GATE_COLS), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    scales = torch.empty(
        (capacity, _SCALE_COLS // 2, 2),
        dtype=torch.float8_e8m0fnu,
        device=hidden_states.device,
    )
    if valid_rows == 0:
        return payload, scales

    value = _situ(hidden_states[:valid_rows], beta, linear_beta)
    valid_payload, scale_bytes = _quantize(value)
    payload[:valid_rows].copy_(valid_payload)
    scales[:valid_rows].copy_(
        scale_bytes.reshape(valid_rows, _SCALE_COLS // 2, 2).view(torch.float8_e8m0fnu)
    )
    return payload, scales
