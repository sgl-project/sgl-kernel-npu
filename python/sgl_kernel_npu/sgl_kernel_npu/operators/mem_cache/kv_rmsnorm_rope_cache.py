"""TorchNPU fused MLA normalization, RoPE, and cache update."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "mem_cache.kv_rmsnorm_rope_cache"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(kv, gamma, cos, sin, index, k_cache, ckv_cache, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_kv_rmsnorm_rope_cache

        _torch_npu_fn = npu_kv_rmsnorm_rope_cache
    return _torch_npu_fn(kv, gamma, cos, sin, index, k_cache, ckv_cache, **kwargs)


@register_operator(_OP_NAME)
def kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    *,
    k_rope_scale=None,
    c_kv_scale=None,
    k_rope_offset=None,
    c_kv_offset=None,
    v=None,
    epsilon=1e-5,
    cache_mode="Norm",
    is_output_kv=False,
    impl="auto",
):
    """Normalize and rotate MLA input and write the specified cache slots.

    kv has layout [batch, 1, seq_len, latent_dim + rope_dim]. gamma scales
    the latent part, and cos/sin rotate the remaining dimensions. The cache
    mode sets the storage layout and interpretation of index. Optional
    scales and offsets are forwarded to the installed TorchNPU.

    Auto selects torch_npu and returns the two updated caches and the two
    backend outputs. The last two outputs are valid when is_output_kv is true.
    """
    optional = {"v": v} if v is not None else {}
    return get_impl_by_name(_OP_NAME, impl)(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        k_rope_scale=k_rope_scale,
        c_kv_scale=c_kv_scale,
        k_rope_offset=k_rope_offset,
        c_kv_offset=c_kv_offset,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=is_output_kv,
        **optional,
    )
