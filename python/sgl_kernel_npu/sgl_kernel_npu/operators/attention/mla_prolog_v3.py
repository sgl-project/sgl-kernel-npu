"""TorchNPU MLA prolog with the V3 calling convention."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_npu_fn = None
_OP_NAME = "attention.mla_prolog_v3"


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(*args, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_mla_prolog_v3

        _torch_npu_fn = npu_mla_prolog_v3
    return _torch_npu_fn(*args, **kwargs)


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
    Returns the five V3 outputs. auto selects torch_npu.
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
