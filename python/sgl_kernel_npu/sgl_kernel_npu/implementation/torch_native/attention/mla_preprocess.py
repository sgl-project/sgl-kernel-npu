"""Independent CPU reference for the fused MLA preprocess operator.

This is an accuracy reference, not a graph-compatible execution fallback. It
follows the PyTorch path that the legacy operator test already used
(``golden2_pytorch``) and adds the physical cache writes so the mutable outputs
have the same meaning as the CANN implementation.

Numerical contract: csrc/mla_preprocess. Intermediate rounding follows the
legacy test: per-tensor quantization casts to FP16 before clamping and rounding,
INT8 GEMM accumulates in INT32 and dequantizes in FP32, and RMSNorm/RoPE compute
in FP32 before casting to the input dtype.
"""

from typing import Optional, Tuple

import torch

K_NOPE = 512
K_PE = 64
Q_RMS = 1536
Q_DIM = 192
Q_NOPE_DIM = 128
MM1_OUT = 2112

_CACHE_TO_INDEX = {"kvcache": 0, "krope_ctkv": 1, "int8_nzcache": 2, "nzcache": 3}


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def _nz_to_nd(weight: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Recover logical [rows, cols] from the 4D int8 FRACTAL_NZ storage.

    The legacy preparation pads the logical matrix to a multiple of 16 rows and
    32 columns, then stores it as [1, cols/32, paddedRows, 32].
    """
    if weight.dim() == 2:
        return _cpu(weight)
    nz = _cpu(weight)
    columns = nz.shape[1]
    padded_rows = nz.shape[2]
    c0 = nz.shape[3]
    logical = (
        nz.reshape(columns, padded_rows // 16, 16, c0)
        .permute(1, 2, 0, 3)
        .reshape(padded_rows, columns * c0)
    )
    return logical[:rows, :cols].contiguous()


def _quant_per_tensor(
    x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor
) -> torch.Tensor:
    x = x.float() / scale.float() + zp.float()
    x = x.to(torch.float16)
    x = torch.clamp(x, -128, 127)
    return torch.round(x).to(torch.int8)


def _quant_per_tensor_muls(
    x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor
) -> torch.Tensor:
    x = x.float() * scale.float() + zp.float()
    x = x.to(torch.float16)
    x = torch.clamp(x, -128, 127)
    return torch.round(x).to(torch.int8)


def _int8_gemm_dequant(
    a_int8: torch.Tensor,
    w_int8: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    y = a_int8.to(torch.int32) @ w_int8.to(torch.int32).t()
    y = y + bias
    if output_dtype == torch.float16:
        # The FP16 path stores a bit-truncated FP32 scale in an INT64 tensor.
        descale = descale.to(torch.int32).view(torch.float32)
    y = y.float() * descale.float()
    return y.to(output_dtype)


def _rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, beta: Optional[torch.Tensor] = None
) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + 1e-6) * gamma.float()
    if beta is not None:
        y = y + beta.float()
    return y


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    return torch.cat([-second_half, first_half], dim=-1)


def _apply_rope_half(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


def _write_linear(
    cache: torch.Tensor, values: torch.Tensor, slots: torch.Tensor
) -> None:
    """Write one logical row per slot into a flat ND cache."""
    flat = cache.reshape(-1, cache.shape[-1])
    flat[slots.long()] = values.to(cache.dtype).reshape(slots.numel(), -1)


def _write_nz(
    cache: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
    c0: int,
) -> None:
    """Write one logical row per slot into the group-major NZ cache layout."""
    groups = values.shape[-1] // c0
    view = cache.view(cache.shape[0], groups, block_size, c0)
    pages = (slots // block_size).long()
    offsets = (slots % block_size).long()
    group = torch.arange(groups, device="cpu")
    view[pages[:, None], group[None, :], offsets[:, None]] = values.to(
        cache.dtype
    ).reshape(slots.numel(), groups, c0)


def mla_preprocess(
    hidden_state: torch.Tensor,
    gamma0: torch.Tensor,
    beta0: torch.Tensor,
    wdqkv: torch.Tensor,
    descale0: torch.Tensor,
    gamma1: torch.Tensor,
    beta1: torch.Tensor,
    wuq: torch.Tensor,
    descale1: torch.Tensor,
    gamma2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wuk: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    slotmapping: torch.Tensor,
    *,
    quant_scale0: Optional[torch.Tensor] = None,
    quant_offset0: Optional[torch.Tensor] = None,
    bias0: Optional[torch.Tensor] = None,
    quant_scale1: Optional[torch.Tensor] = None,
    quant_offset1: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
    ctkv_scale: Optional[torch.Tensor] = None,
    q_nope_scale: Optional[torch.Tensor] = None,
    cache_mode: str = "krope_ctkv",
    quant_mode: str = "per_token_quant_symm",
    q_out0: torch.Tensor,
    kv_cache_out0: torch.Tensor,
    q_out1: torch.Tensor,
    kv_cache_out1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one fused MLA preprocess step and update both paged caches."""
    if quant_mode != "per_tensor_quant_asymm":
        raise NotImplementedError(
            "The reference implements per_tensor_quant_asymm only"
        )
    if cache_mode not in _CACHE_TO_INDEX:
        raise ValueError(f"Unsupported cache_mode: {cache_mode}")
    mode = _CACHE_TO_INDEX[cache_mode]
    dtype = hidden_state.dtype
    device = hidden_state.device
    head_num = wuk.shape[0]
    tokens = hidden_state.shape[0]
    slots = slotmapping.cpu().to(torch.int64)

    hidden = _cpu(hidden_state)
    hidden_dim = hidden_state.shape[-1]
    cos_cpu = _cpu(cos)
    sin_cpu = _cpu(sin)
    wuk_cpu = _cpu(wuk)
    # Weights may arrive in the 4D int8 NZ storage used by the CANN Kernel.
    wdqkv_cpu = _nz_to_nd(wdqkv, MM1_OUT, hidden_dim)
    wuq_cpu = _nz_to_nd(wuq, head_num * Q_DIM, Q_RMS)

    # MM1: quantize the hidden state, then dequantize the INT8 GEMM.
    quant_out = _quant_per_tensor(hidden, _cpu(quant_scale0), _cpu(quant_offset0))
    fused = _int8_gemm_dequant(quant_out, wdqkv_cpu, _cpu(descale0), _cpu(bias0), dtype)
    latent, q = fused.split([K_NOPE + K_PE, Q_RMS], dim=-1)
    k_nope = latent[..., :K_NOPE]
    k_pe = latent[..., K_NOPE:].unsqueeze(1)

    # RMSNorm2 + RMSNorm3.
    q = _rms_norm(q, _cpu(gamma1), _cpu(beta1))
    k_nope = _rms_norm(k_nope, _cpu(gamma2)).unsqueeze(1)

    # MM2.
    quant_out2 = _quant_per_tensor(q, _cpu(quant_scale1), _cpu(quant_offset1))
    q_out = _int8_gemm_dequant(
        quant_out2, wuq_cpu, _cpu(descale1), _cpu(bias1), dtype
    ).view(-1, head_num, Q_DIM)
    q_nope, q_pe = q_out.split([Q_NOPE_DIM, Q_DIM - Q_NOPE_DIM], dim=-1)

    # BMM3: absorb W_uk into Q. FP32 accumulation then cast to the input dtype.
    q_nope_out = (
        torch.bmm(q_nope.transpose(0, 1).float(), wuk_cpu.float())
        .transpose(0, 1)
        .to(dtype)
    )

    # RoPE.
    q_pe_rope = _apply_rope_half(q_pe, cos_cpu, sin_cpu)
    k_pe_rope = _apply_rope_half(k_pe, cos_cpu, sin_cpu)

    if mode == 2:
        q_nope_out = _quant_per_tensor_muls(
            q_nope_out,
            _cpu(q_nope_scale).reshape(1, head_num, 1),
            torch.zeros_like(q_nope_out),
        )
        k_nope_store = _quant_per_tensor(
            k_nope.squeeze(1),
            _cpu(ctkv_scale),
            torch.zeros(tokens, K_NOPE, device="cpu"),
        )
    else:
        q_nope_out = q_nope_out.to(dtype)
        k_nope_store = k_nope.squeeze(1).to(dtype)

    q_pe_rope = q_pe_rope.to(device)
    if mode == 0:
        # One combined 576-wide cache and one 576-wide query output.
        combined_q = torch.cat([q_nope_out.to(dtype), q_pe_rope.cpu()], dim=-1).to(
            device
        )
        q_out0.copy_(combined_q)
        combined_k = torch.cat([k_nope_store, k_pe_rope.squeeze(1)], dim=-1).to(device)
        _write_linear(kv_cache, combined_k, slots)
    else:
        q_out0.copy_(q_nope_out.to(device))
        q_out1.copy_(q_pe_rope)
        k_pe_device = k_pe_rope.squeeze(1).to(device)
        if mode == 1:
            _write_linear(kv_cache, k_nope_store.to(device), slots)
            _write_linear(kv_cache_rope, k_pe_device, slots)
        elif mode == 2:
            _write_nz(kv_cache, k_nope_store.to(device), slots, 128, 32)
            _write_nz(kv_cache_rope, k_pe_device, slots, 128, 16)
        else:
            _write_nz(kv_cache, k_nope_store.to(device), slots, 128, 16)
            _write_nz(kv_cache_rope, k_pe_device, slots, 128, 16)

    return q_out0, kv_cache_out0, q_out1, kv_cache_out1
