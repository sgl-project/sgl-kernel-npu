"""Pack FP8 values, BF16 RoPE bytes and per-group scales on the input device."""

import struct

import torch

from .quantization import _quantize


# Layout dimensions are supplied by the complete packing implementation below.
def _ref_fp8_scales(xf, d, fp8_max):
    """Reference float32 scales for the FP8 path (one per 128-wide block)."""
    quant_col = d - 64
    scale_col = (quant_col + 127) // 128
    bits = 0x3B124925 if fp8_max == 448.0 else 0x37924925
    coeff = struct.unpack("<f", struct.pack("<I", bits))[0]
    scales = torch.zeros(
        (xf.shape[0], scale_col), dtype=torch.float32, device=xf.device
    )
    for j in range(scale_col):
        blk = xf[:, j * 128 : min((j + 1) * 128, quant_col)]
        m = blk.abs().max(dim=1).values
        scales[:, j] = m * coeff
    return scales


def _ref_mxfp8_scales(xf, d, per_group_size, fp8_max, round_scale):
    """Reference E8M0 scale bytes for the MXFP8 path (one per per_group_size group)."""
    scale_col = (d - 64 + per_group_size - 1) // per_group_size
    coeff = struct.unpack("<f", struct.pack("<f", 1.0 / fp8_max))[0]
    out = torch.zeros((xf.shape[0], scale_col), dtype=torch.uint8, device=xf.device)
    for j in range(scale_col):
        blk = xf[:, j * per_group_size : (j + 1) * per_group_size]
        m = blk.abs().max(dim=1).values
        m = torch.clamp(m, min=1e-4)
        m2 = m * coeff  # float32 mul, mirrors Muls(max2, max2, coeff)
        e = (m2.view(torch.int32) >> 23) & 0xFF  # biased exponent of max * coeff
        if round_scale:
            e = e + (e != 0).to(
                torch.int32
            )  # kernel re-bias logic always adds one step
        out[:, j] = e.to(torch.uint8)
    return out


def _check_inputs(cache, x, slots, group, mode, layout):
    def check(ok, message):
        if not ok:
            raise RuntimeError(message)

    check(x.ndim == 2, "x must be 2D tensor")
    check(x.shape[0] > 0 and x.shape[1] > 0, "x dimensions must be positive")
    check(x.dtype == torch.bfloat16, "x must be BF16")
    check(slots.ndim == 1, "slot_mapping must be 1D tensor")
    check(
        slots.shape[0] == x.shape[0], "slot_mapping size must equal x's first dimension"
    )
    check(
        slots.dtype in (torch.int32, torch.int64), "slot_mapping must be INT32 or INT64"
    )
    check(
        cache.dtype in (torch.float8_e5m2, torch.float8_e4m3fn),
        "kv_compress_cache must be FP8_E5M2 or FP8_E4M3FN",
    )
    if cache.ndim == 4:
        check(
            cache.shape[2] == 1,
            "kv_compress_cache 4D tensor requires headnum (dim 2) == 1",
        )
        cache = cache.squeeze(2)
    check(mode in (1, 2), f"quant_mode must be 1 (FP8) or 2 (MXFP8), got {mode}")
    check(layout in (1, 2), f"layout must be 1 or 2, got {layout}")
    if mode == 2:
        check(group in (64, 128), "MXFP8 quant_group_size must be 64 or 128")
    qcols = x.shape[1] - 64
    scales = (qcols + group - 1) // group if mode == 2 else (qcols + 127) // 128
    if layout == 2:
        check(mode == 2, "layout=2 only supports MXFP8")
        check(cache.ndim >= 2, "layout=2 requires kv_compress_cache to be at least 2D")
        check(cache.shape[1] > 0, "blockSize must be positive")
        minimum = cache.shape[1] * (qcols + 128 + (scales + 7) // 8 * 8)
    else:
        used = qcols + 128 + scales * (1 if mode == 2 else 4)
        minimum = (used + 127) // 128 * 128
    check(
        cache.stride(0) <= 0 or cache.stride(0) >= minimum,
        "block_stride is smaller than the packed cache width",
    )


def _native_kv_compress_epilog(
    cache, x, slots, quant_group_size, quant_mode, round_scale_flag, layout
):
    _check_inputs(cache, x, slots, quant_group_size, quant_mode, layout)
    cache = cache.squeeze(2) if cache.ndim == 4 else cache
    device = x.device
    n, d = x.shape
    qcols = d - 64
    group = quant_group_size if quant_mode == 2 else 128
    groups = (qcols + group - 1) // group
    max_value = 448.0 if cache.dtype == torch.float8_e4m3fn else 57344.0
    xf = x.float()
    if quant_mode == 2:
        exponents = _ref_mxfp8_scales(xf, d, group, max_value, round_scale_flag)
        scales = (exponents.to(torch.int32) << 23).view(torch.float32)
    else:
        scales = _ref_fp8_scales(xf, d, max_value)
    quant = torch.empty((n, qcols), dtype=torch.uint8, device=device)
    for j in range(groups):
        block = xf[:, j * group : min((j + 1) * group, qcols)]
        scale = scales[:, j]
        # A zero FP8 group keeps zero values while storing its zero scale.
        normalized = torch.where(scale[:, None] == 0, block, block / scale[:, None])
        quant[:, j * group : min((j + 1) * group, qcols)] = (
            normalized.clamp(-max_value, max_value).to(cache.dtype).view(torch.uint8)
        )
    rope = x[:, -64:].contiguous().view(torch.uint8)
    scale_bytes = (
        ((scales.view(torch.int32) >> 23) & 255).to(torch.uint8)
        if quant_mode == 2
        else scales.view(torch.uint8)
    )
    valid = slots != -1
    indices = slots[valid].long()
    if layout == 2:
        block_size = cache.shape[1]
        value_width = qcols + 128
        scale_width = (groups + 7) // 8 * 8
        values = torch.cat((quant, rope), dim=1)
        scale_rows = torch.zeros((n, scale_width), dtype=torch.uint8, device=device)
        scale_rows[:, :groups] = scale_bytes
        byte = cache.view(torch.uint8).as_strided(
            (cache.shape[0], block_size * (value_width + scale_width)),
            (cache.stride(0), 1),
        )
        block_ids, row_ids = indices // block_size, indices % block_size
        byte[
            block_ids[:, None],
            row_ids[:, None] * value_width + torch.arange(value_width, device=device),
        ] = values[valid]
        byte[
            block_ids[:, None],
            block_size * value_width
            + row_ids[:, None] * scale_width
            + torch.arange(scale_width, device=device),
        ] = scale_rows[valid]
    else:
        used = qcols + 128 + scale_bytes.shape[1]
        width = (used + 127) // 128 * 128
        packed = torch.zeros((n, width), dtype=torch.uint8, device=device)
        if quant_mode == 2:
            packed[:, :128] = rope
            packed[:, 128 : 128 + qcols] = quant
        else:
            packed[:, :qcols] = quant
            packed[:, qcols : qcols + 128] = rope
        packed[:, qcols + 128 : used] = scale_bytes
        stride = cache.stride(0) if layout == 1 else width
        byte = cache.view(torch.uint8).as_strided((cache.shape[0], width), (stride, 1))
        byte[indices] = packed[valid]


def _cann_9_1_e8m0_quantize(x):
    """Independent reference for the deployed A5 9.1 writer's E8M0 rule.

    The original writer/tests use FP32 max*reciprocal after amax>=1e-4,
    then the next binary exponent even when that value is already 2**n.
    This is not the released writer's ceil(log2(scale)) convention.
    """
    values = x.float().cpu()
    amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    raw_scale = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device="cpu")
    _, exponent = torch.frexp(raw_scale)
    scale = torch.ldexp(torch.ones_like(raw_scale), exponent)
    quant = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quant, scale


def _packed_kv_compress_epilog(
    cache,
    x,
    slot_mapping,
    *,
    quant_group_size=64,
    quant_mode="fp8_bf16",
    round_scale=True,
    x_scale=1.0,
):
    if quant_mode not in ("fp8_bf16", "fp8_e8m0") or quant_group_size != 64:
        raise NotImplementedError(
            "reference supports 64-element BF16/E8M0 scale groups"
        )
    cann_9_1 = quant_mode == "fp8_e8m0"
    if cann_9_1 and not round_scale:
        raise NotImplementedError("cann_9_1 E8M0 scales require power-of-two rounding")
    # DSV4 stores 64 bf16 RoPE features followed by group-quantized NoPE and bf16 scales.
    if x.shape[-1] != 512 or cache.shape[-1] != (640 if cann_9_1 else 608):
        raise NotImplementedError(
            "reference requires matching 608B BF16-scale or 640B E8M0-scale rows"
        )
    rows = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    values = x.reshape(-1, 512).float() * x_scale
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        groups = values[i, :448].reshape(7, 64)
        if cann_9_1:
            quant, scales = _cann_9_1_e8m0_quantize(groups)
        else:
            quant, scales = _quantize(
                groups, torch.float8_e4m3fn, round_scale=round_scale
            )
        rows[slot, :128].copy_(
            values[i, 448:].to(torch.bfloat16).contiguous().view(torch.uint8)
        )
        rows[slot, 128:576].copy_(quant.contiguous().view(torch.uint8).flatten())
        if cann_9_1:
            rows[slot, 576:583].copy_(
                (torch.log2(scales).flatten() + 127).to(torch.uint8)
            )
        else:
            rows[slot, 576:590].copy_(
                scales.to(torch.bfloat16).contiguous().view(torch.uint8).flatten()
            )
    return cache


def kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    if isinstance(kwargs["quant_mode"], int):
        return _native_kv_compress_epilog(cache, x, slot_mapping, **kwargs)
    return _packed_kv_compress_epilog(cache, x, slot_mapping, **kwargs)
