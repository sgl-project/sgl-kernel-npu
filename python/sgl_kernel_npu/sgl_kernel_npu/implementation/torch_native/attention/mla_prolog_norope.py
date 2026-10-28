"""Independent, CPU PyTorch reference for the Kimi no-RoPE MLAProlog paths.

This is an accuracy reference, not a graph-compatible execution fallback. NPU
inputs are copied to CPU; outputs and the two updated caches return to their
original devices. No fused MLAProlog or NPU quantization operator is called.

Numerical contract: ops-transformer/attention/mla_prolog_v3/docs/
aclnnMlaPrologV4WeightNz.md. Intermediate rounding and MXFP8 scale semantics are
specified by attention/mla_prolog/op_kernel/arch35/service_rms_norm_arch35.h,
vf/vf_dynamic_quant.h and vf/vf_mul_qr.h in the same upstream repository.
"""

from typing import Optional, Tuple

import torch


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    # A device-to-CPU copy exposes the logical ND shape of an NPU NZ weight.
    return tensor.detach().cpu()


def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 bytes, including its minimum exponent and NaN encoding."""
    bits = _cpu(scale).contiguous().view(torch.uint8).to(torch.int32)
    values = torch.exp2((bits - 127).float())
    return torch.where(bits == 255, float("nan"), values)


def _dequantize_mx(
    value: torch.Tensor, scale: torch.Tensor, *, weight: bool = False
) -> torch.Tensor:
    # Activations use [T, K/32], weights use [N, K/32] for logical [K, N].
    expanded_scale = _decode_e8m0(scale).repeat_interleave(32, dim=-1)
    if weight:
        expanded_scale = expanded_scale.transpose(0, 1)
    return _cpu(value).float() * expanded_scale


def _bf16_round_away(value: torch.Tensor) -> torch.Tensor:
    """BF16 nearest rounding with midpoint ties away (CANN CAST_ROUND)."""
    value = value.float().contiguous()
    bits = value.view(torch.int32)
    # Keep the upper 16 bits, carrying whenever the discarded half is >= 0.5.
    upper = (bits >> 16) + ((bits & 0xFFFF) >= 0x8000).to(torch.int32)
    rounded = (upper << 16).view(torch.float32)
    return torch.where(torch.isfinite(value), rounded, value).to(torch.bfloat16)


def _quantize_mx(
    value: torch.Tensor, scale_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows in groups of 32 using the V4 internal MX rule."""
    value = _bf16_round_away(value)
    groups = value.reshape(*value.shape[:-1], value.shape[-1] // 32, 32)
    # Extract the BF16 exponent instead of log2: zero/subnormal blocks have
    # exponent zero, and exponent 255 propagates to the E8M0 NaN encoding.
    exponents = (groups.contiguous().view(torch.int16).to(torch.int32) >> 7) & 255
    block_exponents = exponents.amax(dim=-1)
    scale_bits = (block_exponents - 8).clamp_min(0)
    scale_bits = torch.where(block_exponents == 255, 255, scale_bits).to(torch.uint8)
    scale = scale_bits.contiguous().view(scale_dtype)
    normalized = groups.float() / _decode_e8m0(scale).unsqueeze(-1)
    normalized = torch.where(block_exponents.unsqueeze(-1) == 0, 0.0, normalized)
    quantized = normalized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.reshape(value.shape), scale


def _rms_norm(
    value: torch.Tensor, gamma: torch.Tensor, epsilon: float, scale: float
) -> torch.Tensor:
    value = value.float()
    variance = value.square().mean(dim=-1, keepdim=True)
    return value / torch.sqrt(variance + epsilon) * _cpu(gamma).float() * scale


def _write_cache(
    cache: torch.Tensor, values: torch.Tensor, slots: torch.Tensor, cache_mode: str
) -> None:
    """Write logical rows while preserving PA_NZ's explicit 32-byte tiles."""
    if cache.numel() == 0 or slots.numel() == 0:
        return
    cache_cpu = _cpu(cache).clone()
    pages, page_size, _, head_dim = cache.shape
    # Byte indexing also works for FP8, whose CPU index_put is not supported on
    # every PyTorch version. These are cache bytes, not NZ weight format tags.
    raw = cache_cpu.view(torch.uint8)
    src = (
        values.to(cache.dtype).contiguous().view(torch.uint8).reshape(slots.numel(), -1)
    )
    # Negative indices are padding tokens in the CANN scatter contract.
    valid = slots >= 0
    src, slots = src[valid], slots[valid]
    if cache_mode == "PA_BSND":
        raw.reshape(pages * page_size, -1)[slots] = src
    else:
        tile_elements = 32 // cache.element_size()
        tiles = head_dim // tile_elements
        raw = raw.reshape(pages, tiles, page_size, 32)
        page = slots // page_size
        offset = slots % page_size
        tile = torch.arange(tiles, device="cpu")
        raw[page[:, None], tile[None, :], offset[:, None]] = src.reshape(
            slots.numel(), tiles, 32
        )
    cache.copy_(cache_cpu.to(cache.device))


def mla_prolog_norope(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    *,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    cache_index: Optional[torch.Tensor] = None,
    dequant_scale_x: Optional[torch.Tensor] = None,
    dequant_scale_w_dq: Optional[torch.Tensor] = None,
    dequant_scale_w_uq_qr: Optional[torch.Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[torch.Tensor] = None,
    quant_scale_ckv: Optional[torch.Tensor] = None,
    quant_scale_ckr: Optional[torch.Tensor] = None,
    smooth_scales_cq: Optional[torch.Tensor] = None,
    actual_seq_len: Optional[torch.Tensor] = None,
    k_nope_clip_alpha: Optional[torch.Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    token_x_dtype: Optional[int] = None,
    weight_dq_dtype: Optional[int] = None,
    weight_uq_qr_dtype: Optional[int] = None,
    weight_dkv_kr_dtype: Optional[int] = None,
    kv_cache_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate no-RoPE mode0/C16 and mode3/C16 or C8, updating both caches."""
    if (weight_quant_mode, kv_cache_quant_mode, query_quant_mode) not in (
        (0, 0, 0),
        (3, 0, 0),
        (3, 1, 1),
    ):
        raise NotImplementedError(
            "The reference supports mode0/C16 and mode3/C16 or C8"
        )
    if (
        cache_mode not in ("PA_BSND", "PA_NZ")
        or ckvkr_repo_mode
        or quant_scale_repo_mode
    ):
        raise NotImplementedError(
            "The reference requires separate PA_BSND or PA_NZ caches"
        )
    if any(
        value is not None
        for value in (
            quant_scale_ckr,
            smooth_scales_cq,
            actual_seq_len,
            k_nope_clip_alpha,
            token_x_dtype,
            weight_dq_dtype,
            weight_uq_qr_dtype,
            weight_dkv_kr_dtype,
            kv_cache_dtype,
        )
    ):
        raise NotImplementedError(
            "Optional quantization variants are outside this reference"
        )
    if cache_index is None:
        raise ValueError("Paged caches require cache_index")
    if any(w.ndim != 2 for w in (weight_dq, weight_uq_qr, weight_dkv_kr)):
        raise NotImplementedError(
            "The reference requires logical two-dimensional weights"
        )

    token_shape = token_x.shape[:-1]
    heads, q_dim, c_dim = weight_uk.shape
    r_dim = weight_dkv_kr.shape[1] - c_dim
    x = _cpu(token_x).reshape(-1, token_x.shape[-1])
    if weight_quant_mode == 3:
        if any(
            scale is None
            for scale in (
                dequant_scale_x,
                dequant_scale_w_dq,
                dequant_scale_w_uq_qr,
                dequant_scale_w_dkv_kr,
            )
        ):
            raise ValueError("MXFP8 requires activation and all three weight scales")
        x = _dequantize_mx(x, dequant_scale_x.reshape(x.shape[0], x.shape[-1] // 32))
        w_dq = _dequantize_mx(weight_dq, dequant_scale_w_dq, weight=True)
        w_uq_qr = _dequantize_mx(weight_uq_qr, dequant_scale_w_uq_qr, weight=True)
        w_dkv_kr = _dequantize_mx(weight_dkv_kr, dequant_scale_w_dkv_kr, weight=True)
    else:
        x = x.float()
        w_dq, w_uq_qr, w_dkv_kr = (
            _cpu(w).float() for w in (weight_dq, weight_uq_qr, weight_dkv_kr)
        )

    cq = x @ w_dq
    ckv_kr = x @ w_dkv_kr
    # Mode0 GEMMs have BF16 destinations; mode3 GEMMs accumulate into FP32.
    if weight_quant_mode == 0:
        cq = cq.to(torch.bfloat16).float()
        ckv_kr = ckv_kr.to(torch.bfloat16).float()
    cq = _rms_norm(cq, rmsnorm_gamma_cq, rmsnorm_epsilon_cq, qc_qr_scale)
    if weight_quant_mode == 3:
        query_norm, qnorm_scale = _quantize_mx(cq, dequant_scale_x.dtype)
        cq = _dequantize_mx(query_norm, qnorm_scale)
    else:
        query_norm = cq.to(torch.bfloat16)
        qnorm_scale = torch.empty(0, dtype=torch.float32, device="cpu")
        cq = query_norm.float()

    qc_qr = (cq @ w_uq_qr).to(torch.bfloat16).reshape(-1, heads, q_dim + r_dim)
    # Absorb W_uk into Q; its input and output are BF16 in both modes.
    query = torch.einsum(
        "tnd,ndc->tnc", qc_qr[..., :q_dim].float(), _cpu(weight_uk).float()
    )
    query = query.to(torch.bfloat16)
    q_r = qc_qr[..., q_dim:].contiguous()
    c = _rms_norm(ckv_kr[..., :c_dim], rmsnorm_gamma_ckv, rmsnorm_epsilon_ckv, kc_scale)
    k_r = ckv_kr[..., c_dim:].to(torch.bfloat16)
    if kv_cache_quant_mode == 1:
        if quant_scale_ckv is None:
            raise ValueError("Static C8 requires quant_scale_ckv")
        c_inv_scale = _cpu(quant_scale_ckv).float().reshape(1)
        c = (c * c_inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_scale = query.float().abs().amax(dim=-1, keepdim=True) * (1.0 / 448.0)
        query = (query.float() / q_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_r = (q_r.float() * (c_inv_scale / q_scale)).to(torch.bfloat16)
    else:
        c = c.to(torch.bfloat16)
        q_scale = torch.empty(0, dtype=torch.float32, device="cpu")

    slots = _cpu(cache_index).reshape(-1).long()
    _write_cache(kv_cache, c, slots, cache_mode)
    _write_cache(kr_cache, k_r, slots, cache_mode)
    query = query.reshape(*token_shape, heads, c_dim)
    q_r = q_r.reshape(*token_shape, heads, r_dim)
    if query_norm_flag:
        query_norm = query_norm.reshape(*token_shape, weight_dq.shape[1])
    else:
        query_norm = torch.empty(0, dtype=weight_uq_qr.dtype, device="cpu")
        qnorm_scale = torch.empty(0, dtype=qnorm_scale.dtype, device="cpu")
    return tuple(
        tensor.to(token_x.device)
        for tensor in (query, q_r, q_scale, query_norm, qnorm_scale)
    )
