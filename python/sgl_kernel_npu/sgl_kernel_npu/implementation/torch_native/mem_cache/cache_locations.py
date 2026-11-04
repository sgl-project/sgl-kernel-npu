"""Packed scatter/gather with PyTorch slice copies and in-place return aliases."""


def _intervals(req_pool_indices, start_offset, end_offset):
    offset = 0
    for row, start, end in zip(
        req_pool_indices.cpu().tolist(),
        start_offset.cpu().tolist(),
        end_offset.cpu().tolist(),
    ):
        yield row, start, end, offset
        offset += end - start


def cache_loc_assign(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    flat = out_cache_loc.view(-1)
    for row, start, end, offset in _intervals(
        req_pool_indices, start_offset, end_offset
    ):
        token_pool[row, start:end].copy_(flat[offset : offset + end - start])
    return token_pool


def cache_loc_update(
    req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc
):
    flat = out_cache_loc.view(-1)
    for row, start, end, offset in _intervals(
        req_pool_indices, start_offset, end_offset
    ):
        flat[offset : offset + end - start].copy_(token_pool[row, start:end])
    return out_cache_loc
