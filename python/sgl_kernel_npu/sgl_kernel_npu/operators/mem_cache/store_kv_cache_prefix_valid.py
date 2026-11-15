"""Public store kv cache prefix valid."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.store_kv_cache_prefix_valid", "auto")
@register_impl("mem_cache.store_kv_cache_prefix_valid", "triton")
def _triton_impl(dst_k, dst_v, src_k, src_v, loc_2d, commit_lens):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.kv_cache_store import (
            store_kv_cache_prefix_valid,
        )

        _triton_fn = store_kv_cache_prefix_valid
    return _triton_fn(dst_k, dst_v, src_k, src_v, loc_2d, commit_lens)


@register_impl("mem_cache.store_kv_cache_prefix_valid", "torch_native")
def _torch_native_impl(dst_k, dst_v, src_k, src_v, loc_2d, commit_lens):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.kv_cache_store import (
            store_kv_cache_prefix_valid,
        )

        _torch_native_fn = store_kv_cache_prefix_valid
    return _torch_native_fn(dst_k, dst_v, src_k, src_v, loc_2d, commit_lens)


@register_operator("mem_cache.store_kv_cache_prefix_valid")
def store_kv_cache_prefix_valid(
    dst_k, dst_v, src_k, src_v, loc_2d, commit_lens, *, impl="auto"
):
    """Commits only each request's accepted prefix from packed K/V source rows to flattened cache slots. The K and V head dimensions may differ. Unaccepted source locations are not dereferenced and other cache rows remain unchanged.

    Args:
        dst_k (Tensor): Writable [slots,heads,Kdim] cache, contiguous head/dim axes. Default: Required.
        dst_v (Tensor): Writable [slots,heads,Vdim] cache, same dtype as dst_k. Default: Required.
        src_k (Tensor): Packed [batch*width,heads,Kdim] keys, same dtype as caches. Default: Required.
        src_v (Tensor): Packed [batch*width,heads,Vdim] values, same dtype as caches. Default: Required.
        loc_2d (Tensor): Integer [batch,width] destination slots. Accepted positions must be valid; ignored positions may contain sentinels. Default: Required.
        commit_lens (Tensor): Integer [batch] accepted prefix sizes in [0,width]. Default: Required.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        Writes selected dst_k/dst_v rows in place.
    """
    return get_impl_by_name("mem_cache.store_kv_cache_prefix_valid", impl)(
        dst_k, dst_v, src_k, src_v, loc_2d, commit_lens
    )
