"""Public packed cache-location scatter."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._cache_loc_checks import check_cache_loc_inputs


@register_impl("mem_cache.cache_loc_assign", "auto")
@register_impl("mem_cache.cache_loc_assign", "ascendc")
def _ascendc_impl(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    import torch

    return torch.ops.npu.cache_loc_assign(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )


@register_impl("mem_cache.cache_loc_assign", "torch_native")
def _torch_native_impl(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    from ...implementation.torch_native.mem_cache.cache_locations import (
        cache_loc_assign,
    )

    return cache_loc_assign(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )


@register_operator("mem_cache.cache_loc_assign")
def cache_loc_assign(
    req_pool_indices,
    token_pool,
    start_offset,
    end_offset,
    out_cache_loc,
    *,
    impl="auto"
):
    """Scatter packed locations to token_pool in place; return token_pool.

    Request indices are int32/int64, offsets int64, and pool/locations int32.
    AscendC requires MAX_STEP=16 and compatible storage/UB capacity.
    auto selects ascendc without fallback; torch_native is independent PyTorch.
    """
    check_cache_loc_inputs(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )
    return get_impl_by_name("mem_cache.cache_loc_assign", impl)(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )
