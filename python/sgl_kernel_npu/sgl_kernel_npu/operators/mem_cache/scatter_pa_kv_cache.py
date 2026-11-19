"""TorchNPU scatter into paged key and value caches."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "mem_cache.scatter_pa_kv_cache"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(key, value, key_cache, value_cache, slot_mapping, **kwargs):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_scatter_pa_kv_cache

        _torch_npu_fn = npu_scatter_pa_kv_cache
    return _torch_npu_fn(key, value, key_cache, value_cache, slot_mapping, **kwargs)


@register_operator(_OP_NAME)
def scatter_pa_kv_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    *,
    compress_lens=None,
    compress_seq_offsets=None,
    seq_lens=None,
    cache_mode="PA_NZ",
    impl="auto",
):
    """Scatter K/V into caller-owned paged caches and return None.

    For uncompressed ND caches, pass cache_mode="Norm", K/V tensors of
    shape [tokens, heads, dim] and slot_mapping of shape [tokens]. The
    optional lengths and offsets describe compressed-cache input.
    Auto selects torch_npu.
    """
    return get_impl_by_name(_OP_NAME, impl)(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        compress_lens=compress_lens,
        compress_seq_offsets=compress_seq_offsets,
        seq_lens=seq_lens,
        cache_mode=cache_mode,
    )
