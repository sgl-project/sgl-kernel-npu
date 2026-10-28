"""Public CANN MLAProlog V4 entry point and independent numerical reference."""

from typing import Optional, Tuple

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None
_torch_native_fn = None


@register_impl(
    "attention.mla_prolog", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "attention.mla_prolog", "cann", supported_generations={Generation.ARCH35}
)
def _cann_impl(*args, **kwargs):
    global _cann_fn
    if _cann_fn is None:
        from cann_ops_transformer import mla_prolog

        _cann_fn = mla_prolog
    return _cann_fn(*args, **kwargs)


@register_impl(
    "attention.mla_prolog", "torch_native", supported_generations={Generation.ARCH35}
)
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.mla_prolog import mla_prolog

        _torch_native_fn = mla_prolog
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.mla_prolog")
def mla_prolog(
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
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project MLA queries and update both caches in place; auto selects CANN V4.

    The CANN backend preserves the public ``cann_ops_transformer.mla_prolog``
    contract. The independent reference covers no-RoPE BF16 and MXFP8
    projections with C16 cache, and MXFP8 projections with static C8 cache.

    Args:
        token_x: [T, He] or [B, S, He] tokens (BF16 or MXFP8).
        weight_dq: [He, Hcq] down projection, in CANN FRACTAL_NZ storage.
        weight_uq_qr: [Hcq, N * (D + Dr)] query projection, FRACTAL_NZ.
        weight_uk: [N, D, Hckv] BF16 key absorption weights.
        weight_dkv_kr: [He, Hckv + Dr] cache projection, FRACTAL_NZ.
        rmsnorm_gamma_cq: [Hcq] query RMSNorm weights.
        rmsnorm_gamma_ckv: [Hckv] cache RMSNorm weights.
        kv_cache: Mutable latent cache, BF16 (C16) or FP8 E4M3 (C8).
        kr_cache: Mutable BF16 side cache; no-RoPE still has side features.
        rope_sin, rope_cos: Both None disables RoPE; otherwise CANN RoPE inputs.
        cache_index: Int32/int64 token slot indices for the paged cache.
        dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
            dequant_scale_w_dkv_kr: Input/projection dequantization scales.
        quant_scale_ckv: Static C8 quantization multiplier (inverse scale).
        quant_scale_ckr, smooth_scales_cq, actual_seq_len, k_nope_clip_alpha:
            Optional CANN inputs, forwarded unchanged.
        rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv: RMSNorm epsilon values.
        cache_mode: CANN cache layout; the reference covers ``PA_BSND`` and
            ``PA_NZ``.
        query_norm_flag: Whether to return the intermediate normalized query.
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode:
            CANN quantization modes; this contribution covers (0,0,0),
            (3,0,0) and (3,1,1).
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale:
            CANN storage/scaling attributes, forwarded unchanged.
        token_x_dtype, weight_dq_dtype, weight_uq_qr_dtype,
            weight_dkv_kr_dtype, kv_cache_dtype: Optional CANN dtype overrides.
        impl: ``auto`` (CANN), ``cann`` or ``torch_native``.

    Returns:
        query, query_rope, dequant_scale_q_nope, query_norm,
        dequant_scale_q_norm. Disabled outputs remain empty tensors. The two
        cache inputs are updated, not returned. See the API documentation for
        mode-dependent output shapes and dtypes.
    """
    impl_fn = get_impl_by_name("attention.mla_prolog", impl)
    return impl_fn(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        kv_cache,
        kr_cache,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        cache_index=cache_index,
        dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        quant_scale_ckv=quant_scale_ckv,
        quant_scale_ckr=quant_scale_ckr,
        smooth_scales_cq=smooth_scales_cq,
        actual_seq_len=actual_seq_len,
        k_nope_clip_alpha=k_nope_clip_alpha,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode=cache_mode,
        query_norm_flag=query_norm_flag,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode,
        ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        qc_qr_scale=qc_qr_scale,
        kc_scale=kc_scale,
        token_x_dtype=token_x_dtype,
        weight_dq_dtype=weight_dq_dtype,
        weight_uq_qr_dtype=weight_uq_qr_dtype,
        weight_dkv_kr_dtype=weight_dkv_kr_dtype,
        kv_cache_dtype=kv_cache_dtype,
    )
