"""TorchNPU paged-cache writer used by the NPU attention KV pool."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "mem_cache.reshape_and_cache"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(key, value, key_cache, value_cache, slot_indices):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import _npu_reshape_and_cache

        _torch_npu_fn = _npu_reshape_and_cache
    return _torch_npu_fn(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=slot_indices,
    )


@register_operator(_OP_NAME)
def reshape_and_cache(key, value, key_cache, value_cache, slot_indices, *, impl="auto"):
    """Write K/V token rows into paged cache tensors at slot_indices in place.

    K/V caches in the attention pool have shape [pages, page_size, heads, dim].
    Auto selects torch_npu. The operation returns None.
    """
    return get_impl_by_name(_OP_NAME, impl)(
        key, value, key_cache, value_cache, slot_indices
    )
