def get_autotune_config():
    return [
        triton.Config({}, multibuffer=True)
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=[],
)
@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    start = start.to(tl.float32)
    end = end.to(tl.float32)
    out_offset = tl.sum(end - start, axis=0)
    out_offset = out_offset.to(tl.int64)

    out_cache_ptr = out_cache_loc + out_offset

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        load_offset = kv_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        save_offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
