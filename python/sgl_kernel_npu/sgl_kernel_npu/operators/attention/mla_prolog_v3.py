"""TorchNPU MLA prolog with the V3 calling convention."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "attention.mla_prolog_v3"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-transformer")
def _torch_npu_impl(*args, **kwargs):
    return torch_npu.npu_mla_prolog_v3(*args, **kwargs)


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.attention.mla_prolog_v3 import mla_prolog_v3

        _native_impl = mla_prolog_v3
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
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
    impl="auto",
    **kwargs,
):
    """Project and normalize MLA queries and update the supplied caches in place.

    The V3 SDK accepts token activations, Q/KV projection weights, normalization
    weights, rotary tensors, and the latent and rotary caches. cache_index,
    quantization attributes and other optional SDK keywords pass through.
    Returns the five V3 outputs. auto selects ops-transformer.
    """
    return get_impl_by_name(_OP_NAME, impl)(
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
        **kwargs,
    )
