"""Public MLA preprocess dispatch for the independent attention CANN operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_ARCH22 = {Generation.ARCH22}


@register_impl("attention.mla_preprocess", "auto", supported_generations=_ARCH22)
@register_impl("attention.mla_preprocess", "ascendc", supported_generations=_ARCH22)
def _ascendc_impl(
    hidden_state,
    gamma0,
    beta0,
    wdqkv,
    descale0,
    gamma1,
    beta1,
    wuq,
    descale1,
    gamma2,
    cos,
    sin,
    wuk,
    kv_cache,
    kv_cache_rope,
    slotmapping,
    *,
    quant_scale0,
    quant_offset0,
    bias0,
    quant_scale1,
    quant_offset1,
    bias1,
    ctkv_scale=None,
    q_nope_scale=None,
    cache_mode="krope_ctkv",
    quant_mode="per_tensor_quant_asymm",
    q_out0,
    kv_cache_out0,
    q_out1,
    kv_cache_out1,
):
    return torch.ops.npu.mla_preprocess(
        hidden_state,
        gamma0,
        beta0,
        wdqkv,
        descale0,
        gamma1,
        beta1,
        wuq,
        descale1,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
        slotmapping,
        quant_scale0,
        quant_offset0,
        bias0,
        quant_scale1,
        quant_offset1,
        bias1,
        ctkv_scale=ctkv_scale,
        q_nope_scale=q_nope_scale,
        cache_mode=cache_mode,
        quant_mode=quant_mode,
        q_out0=q_out0,
        kv_cache_out0=kv_cache_out0,
        q_out1=q_out1,
        kv_cache_out1=kv_cache_out1,
    )


@register_impl("attention.mla_preprocess", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.mla_preprocess import (
            mla_preprocess as torch_native_mla_preprocess,
        )

        _torch_native_fn = torch_native_mla_preprocess
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.mla_preprocess")
def mla_preprocess(
    hidden_state,
    gamma0,
    beta0,
    wdqkv,
    descale0,
    gamma1,
    beta1,
    wuq,
    descale1,
    gamma2,
    cos,
    sin,
    wuk,
    kv_cache,
    kv_cache_rope,
    slotmapping,
    *,
    quant_scale0,
    quant_offset0,
    bias0,
    quant_scale1,
    quant_offset1,
    bias1,
    ctkv_scale=None,
    q_nope_scale=None,
    cache_mode="krope_ctkv",
    quant_mode="per_tensor_quant_asymm",
    q_out0,
    kv_cache_out0,
    q_out1,
    kv_cache_out1,
    impl="auto",
):
    """Fuse MLA RMSNorm, quantized GEMM, RoPE and paged-cache writes.

    ``auto`` selects ``ascendc`` without fallback. The four output/cache tensors
    are caller-owned and written in place; they are also returned, in the same
    order as the arguments. ``cache_mode`` accepts ``krope_ctkv``,
    ``int8_nzcache`` and ``nzcache``.
    """
    return get_impl_by_name("attention.mla_preprocess", impl)(
        hidden_state,
        gamma0,
        beta0,
        wdqkv,
        descale0,
        gamma1,
        beta1,
        wuq,
        descale1,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
        slotmapping,
        quant_scale0=quant_scale0,
        quant_offset0=quant_offset0,
        bias0=bias0,
        quant_scale1=quant_scale1,
        quant_offset1=quant_offset1,
        bias1=bias1,
        ctkv_scale=ctkv_scale,
        q_nope_scale=q_nope_scale,
        cache_mode=cache_mode,
        quant_mode=quant_mode,
        q_out0=q_out0,
        kv_cache_out0=kv_cache_out0,
        q_out1=q_out1,
        kv_cache_out1=kv_cache_out1,
    )
