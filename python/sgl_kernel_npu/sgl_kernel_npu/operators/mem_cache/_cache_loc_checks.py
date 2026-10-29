"""Common dtype and contiguous-output checks for cache-location operators."""

import torch


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
