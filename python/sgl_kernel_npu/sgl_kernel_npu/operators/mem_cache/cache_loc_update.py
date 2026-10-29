"""Public packed cache-location gather."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._cache_loc_checks import check_cache_loc_inputs


@register_impl("mem_cache.cache_loc_update", "auto")
@register_impl("mem_cache.cache_loc_update", "ascendc")
def _ascendc_impl(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    import torch

    return torch.ops.npu.cache_loc_update(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )


@register_impl("mem_cache.cache_loc_update", "torch_native")
def _torch_native_impl(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    from ...implementation.torch_native.mem_cache.cache_locations import (
        cache_loc_update,
    )

    return cache_loc_update(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )


@register_operator("mem_cache.cache_loc_update")
def cache_loc_update(
    req_pool_indices,
    token_pool,
    start_offset,
    end_offset,
    out_cache_loc,
    *,
    impl="auto"
):
    """Gather token_pool intervals into packed locations in place; return out_cache_loc.

    Request indices are int32/int64, offsets int64, and pool/locations int32.
    AscendC requires MAX_STEP=16 and compatible storage/UB capacity.
    auto selects ascendc without fallback; torch_native is independent PyTorch.
    """
    check_cache_loc_inputs(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )
    return get_impl_by_name("mem_cache.cache_loc_update", impl)(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )
