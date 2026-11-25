"""Reference for the unquantized, separate-cache MLA Prolog V3 path."""

import torch

from ..utils.mla_prolog import mlaprolog_v4


def mla_prolog_v3(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    rope_sin,
    rope_cos,
    kv_cache,
    kr_cache,
    *,
    weight_quant_mode=0,
    kv_cache_quant_mode=0,
    query_quant_mode=0,
    query_norm_flag=False,
    **kwargs
):
    if token_x.dtype != torch.bfloat16 or (
        weight_quant_mode,
        kv_cache_quant_mode,
        query_quant_mode,
    ) != (0, 0, 0):
        raise NotImplementedError(
            "torch_native supports unquantized BF16 MLA Prolog V3"
        )
    # V3 and V4 share the BF16 projection, normalization, rotation and paged
    # cache write when all quantization modes are disabled.
    query, query_rope, _, normalized, _ = mlaprolog_v4(
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
        query_norm_flag=query_norm_flag,
        **kwargs
    )
    return query, query_rope, None, normalized if query_norm_flag else None, None
