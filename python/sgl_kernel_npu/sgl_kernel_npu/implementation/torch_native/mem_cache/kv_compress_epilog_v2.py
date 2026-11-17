"""Independent reference for the MXFP8/MXFP4 quantizing KV writer.

Mirrors the released cann-recipes-infer ``KvCompressEpilogV2`` kernel bit for
bit on finite inputs: per-group scales (rounded to the next power of two for
MXFP8, OCP MX exponent scaling for MXFP4), round-to-nearest-even FP8/FP4
casts, ``[values | scales]`` cache rows and paged 4D addressing. Only the
quantized payload regions are rewritten; untouched cache bytes are preserved.
"""

import torch

_FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}
_MXFP8_GROUP = 32
# e2m1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}; RINT rounds ties to even, so
# each boundary records whether an exact tie rounds up to the next code.
_MXFP4_BOUNDARIES = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)
_BF16_EXP_BIAS = 127
_BF16_NAN_BITS = 0x7F81
_BF16_INV_BIAS_FIELD = 0xFE  # exponent field of the kernel's INV_BIAS constant
_FP4_MAX_EXP_FIELD = 0x02  # exponent field of the kernel's fp4MaxExp constant
_FP4_SPECIAL_EXP = 0xFF
_ALIGN = 32


def _normalize_mode(quant_mode):
    mode = quant_mode.strip().lower()
    if mode not in ("mxfp8_bf16", "mxfp4_bf16"):
        raise ValueError(
            "quant_mode should be one of [mxfp8_bf16, mxfp4_bf16], but got "
            f"'{quant_mode}'"
        )
    return mode


def _scale_words_to_bytes(bits):
    """Split uint16 scale words into little-endian byte columns."""
    bits = bits.to(torch.int64)
    low = (bits & 0xFF).to(torch.uint8)
    high = ((bits >> 8) & 0xFF).to(torch.uint8)
    return torch.stack((low, high), dim=-1).reshape(bits.shape[0], -1)


def _ceil_power_of_two(scale):
    """Kernel's roundScale: 2^ceil(log2(scale)) via the float32 bit pattern."""
    bits = scale.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    unbiased = exponent - _BF16_EXP_BIAS + (mantissa != 0).to(torch.int32)
    return ((unbiased + _BF16_EXP_BIAS) << 23).view(torch.float32)


def _quantize_mxfp8(xf, cache_dtype, round_scale):
    """Return value bytes [n, d] and scale words [n, d/32] for MXFP8."""
    fp8_max = _FP8_MAX[cache_dtype]
    rows, dim = xf.shape
    grouped = xf.view(rows, -1, _MXFP8_GROUP)
    amax = grouped.abs().amax(-1, keepdim=True)
    # Kernel: Maxs(amax, 1e-4) then Muls(scale, scale, 1/max) in float32.
    coeff = torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=xf.device)
    scale = torch.clamp(amax, min=1e-4) * coeff
    if round_scale:
        scale = _ceil_power_of_two(scale)
    quant = grouped / scale
    quant = quant.clamp(-fp8_max, fp8_max).to(cache_dtype).view(torch.uint8)
    groups = scale.shape[1]
    scale_bits = scale.view(rows, groups).to(torch.bfloat16).view(torch.uint16)
    return quant.view(rows, dim), scale_bits


def _quantize_mxfp4(xf, group):
    """Return packed-nibble bytes [n, d/2] and scale words [n, d/group].

    Scale = 2^(floor(log2(amax)) - 2) per OCP MX with emax=2 for e2m1, stored
    as its bf16 bit pattern; groups below the subnormal clamp store scale word
    0 with a 2^127 multiplier, matching the kernel's fp4MaxExp clamp.
    """
    rows, dim = xf.shape
    grouped = xf.view(rows, -1, group)
    amax = grouped.abs().amax(-1, keepdim=True)
    biased = (amax.view(torch.int32) >> 23) & 0xFF
    clamped = biased <= _FP4_MAX_EXP_FIELD
    scale_field = torch.where(clamped, torch.zeros_like(biased), biased - 2)
    scale_bits = scale_field << 7
    scale_bits = torch.where(
        biased == _FP4_SPECIAL_EXP,
        torch.full_like(scale_bits, _BF16_NAN_BITS),
        scale_bits,
    )
    inverse_field = _BF16_INV_BIAS_FIELD - scale_field
    # Already a float32 exponent field: 2^(127 - scale_field), no bias added.
    inverse = (inverse_field << 23).view(torch.float32)
    scaled = (grouped * inverse).view(rows, -1)
    magnitudes = scaled.abs()
    codes = torch.zeros_like(scaled, dtype=torch.int64)
    for boundary, ties_up in _MXFP4_BOUNDARIES:
        codes += (magnitudes >= boundary) if ties_up else (magnitudes > boundary)
    # e2m1 sign bit (nibble bit 3); signbit covers -0.0 and negatives that
    # round to zero magnitude, matching the kernel.
    codes |= torch.signbit(scaled).to(torch.int64) << 3
    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    data = (even | (odd << 4)).to(torch.uint8)
    return data, scale_bits.view(rows, -1)


def kv_compress_epilog_v2(
    cache,
    x,
    slot_mapping,
    *,
    quant_group_size=32,
    quant_mode="mxfp8_bf16",
    round_scale=True,
    x_scale=1.0,
):
    """Quantize bf16 rows in place into the paged KV cache; return cache.

    Row layout: [values | uint16 scale words] (MXFP4 packs two e2m1 codes per
    byte); 2D rows are zero-padded to 32 bytes while 4D rows stay compact at
    ``tokenStride`` inside each block page. Slots outside ``[0, cache_rows)``
    are skipped, matching the kernel.
    """
    mode = _normalize_mode(quant_mode)
    device = x.device
    rows, dim = x.shape
    if mode == "mxfp4_bf16":
        data, scale_bits = _quantize_mxfp4(x.to(torch.float32), quant_group_size)
    else:
        data, scale_bits = _quantize_mxfp8(
            x.to(torch.float32), cache.dtype, round_scale
        )
    data_width = data.shape[1]
    token_width = data_width + scale_bits.shape[1] * 2
    kv_width = (token_width + _ALIGN - 1) // _ALIGN * _ALIGN
    ids = slot_mapping.to(torch.int64)

    byte = cache.view(torch.uint8)
    if cache.ndim == 2:
        packed = torch.zeros((rows, kv_width), dtype=torch.uint8, device=device)
        packed[:, :data_width] = data
        packed[:, data_width:token_width] = _scale_words_to_bytes(scale_bits)
        flat = byte.as_strided((cache.shape[0], cache.shape[1]), (cache.stride(0), 1))
        valid = (ids >= 0) & (ids < cache.shape[0])
        flat[ids[valid], :kv_width] = packed[valid]
        return cache

    blocks, block_size, _, width = cache.shape
    flat = byte.as_strided((blocks, block_size * width), (cache.stride(0), 1))
    # 4D rows stay compact inside each block page: values then scales.
    packed = torch.cat((data, _scale_words_to_bytes(scale_bits)), dim=1)
    valid = (ids >= 0) & (ids < blocks * block_size)
    selected = ids[valid]
    columns = (selected % block_size)[:, None] * token_width + torch.arange(
        token_width, device=device
    )
    flat[(selected // block_size)[:, None], columns] = packed[valid]
    return cache
