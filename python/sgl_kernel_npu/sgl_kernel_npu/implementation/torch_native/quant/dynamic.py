"""CPU PyTorch references for per-token, block and MX dynamic quantization."""

import torch
import torch.nn.functional as F

_FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}
_FP4_DTYPES = (torch.float4_e2m1fn_x2, torch.uint8)


def _scale_and_normalize(values, maximum, max_value):
    scale = maximum / max_value
    divisor = torch.where(scale == 0, 1.0, scale)
    return scale, (values / divisor).clamp(-max_value, max_value)


def dynamic_quant(
    input,
    *,
    smooth_scales=None,
    group_index=None,
    dst_type=None,
    quant_mode="pertoken",
    dst_type_max=0.0,
):
    dst_type = torch.int8 if dst_type is None else dst_type
    if (
        dst_type not in (torch.int8, *_FP8_MAX)
        or quant_mode != "pertoken"
        or dst_type_max != 0
    ):
        raise NotImplementedError(
            "torch_native supports per-token INT8/FP8 quantization"
        )
    x = input.detach().cpu().float()
    if group_index is not None:
        if smooth_scales is None or x.ndim != 2:
            raise ValueError(
                "Grouped smoothing requires 2D input and per-expert scales"
            )
        boundaries = group_index.detach().cpu().long()
        sizes = torch.diff(boundaries, prepend=boundaries.new_zeros(1))
        smoothing = smooth_scales.detach().cpu().repeat_interleave(sizes, dim=0)
        x = x * smoothing.float()
    elif smooth_scales is not None:
        x = x * smooth_scales.detach().cpu().float()
    scale, normalized = _scale_and_normalize(
        x, x.abs().amax(dim=-1, keepdim=True), _FP8_MAX.get(dst_type, 127.0)
    )
    if dst_type == torch.int8:
        normalized = normalized.round()
    return _to_device(normalized.to(dst_type), input.device), scale.squeeze(-1).to(
        input.device
    )


def dynamic_block_quant(
    x,
    *,
    min_scale=0.0,
    round_mode="rint",
    dst_type=torch.int8,
    row_block_size=1,
    col_block_size=128,
    dst_type_max=0.0,
):
    if (
        dst_type not in (torch.int8, *_FP8_MAX)
        or round_mode != "rint"
        or dst_type_max != 0
        or min_scale != 0
    ):
        raise NotImplementedError(
            "torch_native supports RINT INT8/FP8 block quantization with default scale limits"
        )
    x_cpu = x.detach().cpu().float()
    rows, cols = x_cpu.shape[-2:]
    block_rows, block_cols = row_block_size, col_block_size
    padded = F.pad(x_cpu, (0, -cols % block_cols, 0, -rows % block_rows))
    blocks = padded.unflatten(-2, (-1, block_rows)).unflatten(-1, (-1, block_cols))
    scale, normalized = _scale_and_normalize(
        blocks,
        blocks.abs().amax(dim=(-3, -1), keepdim=True),
        _FP8_MAX.get(dst_type, 127.0),
    )
    if dst_type == torch.int8:
        normalized = normalized.round()
    values = normalized.reshape(padded.shape)[..., :rows, :cols].to(dst_type)
    return _to_device(values, x.device), scale.squeeze(-1).squeeze(-2).to(x.device)


def _to_device(tensor, device):
    # Use a byte transfer for shell dtypes without generic NPU copy kernels.
    if tensor.element_size() == 1:
        return tensor.contiguous().view(torch.uint8).to(device).view(tensor.dtype)
    return tensor.to(device)


def _mx_scale(blocks, max_exponent):
    magnitude = blocks.abs().amax(dim=-1)
    finite = torch.isfinite(magnitude)
    _, exponent = torch.frexp(magnitude)
    bits = (exponent - 1 - max_exponent + 127).clamp(0, 254)
    bits = torch.where(magnitude == 0, 0, bits)
    bits = torch.where(finite, bits, 255).to(torch.uint8)
    # A float64 exponentiation also covers subnormal E8M0 scales in FP32.
    scale = torch.exp2((bits.to(torch.float64) - 127)).float()
    normalized = blocks / scale.unsqueeze(-1)
    normalized = torch.where(finite.unsqueeze(-1), normalized, torch.nan)
    return normalized, bits


def _paired_scales(bits):
    bits = F.pad(bits, (0, bits.size(-1) % 2))
    return bits.unflatten(-1, (-1, 2))


def _pack_e2m1(values):
    # A boundary tie advances only if the lower magnitude code is odd.
    thresholds = values.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    magnitude = values.abs().nan_to_num(nan=0)
    lower = torch.bucketize(magnitude.contiguous(), thresholds)
    tied = magnitude == thresholds[lower.clamp_max(len(thresholds) - 1)]
    code = lower + (tied & ((lower & 1) == 1))
    code = (code | (torch.signbit(values).long() << 3)).to(torch.uint8)
    return code[..., ::2] | (code[..., 1::2] << 4)


def dynamic_mx_quant(
    input,
    *,
    axis=-1,
    round_mode="rint",
    dst_type=torch.float4_e2m1fn_x2,
    block_size=32,
    scale_alg=0,
    dst_type_max=0.0,
    max_low_bound=0.0,
):
    if (
        axis not in (-1, input.ndim - 1)
        or round_mode != "rint"
        or scale_alg not in (None, 0)
        or dst_type_max != 0
        or max_low_bound != 0
        or dst_type not in (*_FP8_MAX, *_FP4_DTYPES)
    ):
        raise NotImplementedError("torch_native supports last-axis MX RINT algorithm 0")
    width = input.size(-1)
    if dst_type in _FP4_DTYPES and width % 2:
        raise ValueError("Packed FP4 requires an even input width")
    x = input.detach().cpu().float()
    blocks = F.pad(x, (0, -width % block_size)).unflatten(-1, (-1, block_size))
    max_exponent = {torch.float8_e4m3fn: 8, torch.float8_e5m2: 15}.get(dst_type, 2)
    values, bits = _mx_scale(blocks, max_exponent)
    values = values.flatten(-2)[..., :width]
    if dst_type in _FP4_DTYPES:
        quantized = _pack_e2m1(values)
    else:
        limit = _FP8_MAX[dst_type]
        quantized = values.clamp(-limit, limit).to(dst_type)
    scales = _paired_scales(bits)
    # OpPlugin returns byte carriers for the FP4 payload and all E8M0 scales.
    return _to_device(quantized, input.device), scales.to(input.device)


def dynamic_dual_level_mx_quant(input, *, smooth_scale=None, round_mode="rint"):
    if smooth_scale is not None or round_mode != "rint":
        raise NotImplementedError("torch_native supports unsmoothed dual-level RINT")
    width = input.size(-1)
    if width % 2:
        raise ValueError("Packed FP4 requires an even input width")
    x = input.detach().cpu().float()
    level0_blocks = F.pad(x, (0, -width % 512)).unflatten(-1, (-1, 512))
    level0, normalized = _scale_and_normalize(
        level0_blocks, level0_blocks.abs().amax(dim=-1, keepdim=True), 6.0
    )
    intermediate = normalized.to(input.dtype).float().flatten(-2)[..., :width]
    values, bits = _mx_scale(
        F.pad(intermediate, (0, -width % 32)).unflatten(-1, (-1, 32)), 2
    )
    packed = _pack_e2m1(values.flatten(-2)[..., :width])
    return (
        packed.to(input.device),
        level0.squeeze(-1).to(input.device),
        _paired_scales(bits).to(input.device),
    )
