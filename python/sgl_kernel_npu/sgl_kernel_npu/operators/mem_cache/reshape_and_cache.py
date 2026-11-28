"""TorchNPU paged-cache writer used by the NPU attention KV pool."""

import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "mem_cache.reshape_and_cache"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ascend-transformer-boost")
def _torch_npu_impl(key, value, key_cache, value_cache, slot_indices):
    return torch_npu._npu_reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=slot_indices,
    )


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.mem_cache.paged_cache import (
            reshape_and_cache,
        )

        _native_impl = reshape_and_cache
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def reshape_and_cache(key, value, key_cache, value_cache, slot_indices, *, impl="auto"):
    """Write K/V token rows into paged cache tensors at slot_indices in place.

    K/V caches in the attention pool have shape [pages, page_size, heads, dim].
    Auto selects ascend-transformer-boost. The operation returns None.
    """
    return get_impl_by_name(_OP_NAME, impl)(
        key, value, key_cache, value_cache, slot_indices
    )
