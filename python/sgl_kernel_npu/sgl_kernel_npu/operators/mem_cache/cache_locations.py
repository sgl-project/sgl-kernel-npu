"""Public packed cache-location scatter and gather."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


def check_cache_loc_inputs(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    if not out_cache_loc.is_contiguous():
        raise RuntimeError("outCacheLoc must be contiguous")
    if (
        req_pool_indices.dtype not in (torch.int32, torch.int64)
        or token_pool.dtype != torch.int32
        or start_offset.dtype != torch.int64
        or end_offset.dtype != torch.int64
        or out_cache_loc.dtype != torch.int32
    ):
        raise ValueError(
            "Only support inputTensor combo1: int64, int32, int64, int64, int32; "
            "combo2: int32, int32, int64, int64, int32"
        )


@register_impl("mem_cache.cache_loc_assign", "auto")
@register_impl("mem_cache.cache_loc_assign", "ascendc")
def _ascendc_assign(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    import torch

    return torch.ops.npu.cache_loc_assign(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )


@register_impl("mem_cache.cache_loc_assign", "torch_native")
def _torch_native_assign(
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


@register_impl("mem_cache.cache_loc_update", "auto")
@register_impl("mem_cache.cache_loc_update", "ascendc")
def _ascendc_update(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    import torch

    return torch.ops.npu.cache_loc_update(
        req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
    )


@register_impl("mem_cache.cache_loc_update", "torch_native")
def _torch_native_update(
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
