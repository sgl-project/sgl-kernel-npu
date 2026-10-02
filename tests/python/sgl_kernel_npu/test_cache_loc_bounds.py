"""cache_loc_assign / cache_loc_update must stay inside out_cache_loc.

The host used to size the kernel's view of out_cache_loc as batch * MAX_STEP
int32 rather than the tensor's own length, so the kernel loaded that many and
stored them all back. The store puts back exactly what the load read, so the
overrun never shows up in the values -- it only surfaces when the bytes past the
tensor are unmapped, as an MTE invalid GM address ("vector core exception").

The first case therefore pins an exactly sized out_cache_loc at the end of a
device block of its own, so that anything the kernel touches past the tensor
leaves the block. The second case is the plain accuracy check.
"""

import sgl_kernel_npu
import torch
import torch_npu

MAX_STEP = 16  # csrc/cache_location_assign/op_host/tiling/cache_loc_assign.h
POOL_ROWS = 512
ROW_LEN = 8192


def make_case(bs, step, req_dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    req_to_token = torch.randint(
        0, 1 << 20, (POOL_ROWS, ROW_LEN), generator=gen, dtype=torch.int32
    ).npu()
    req_pool_indices = torch.randperm(POOL_ROWS, generator=gen)[:bs].to(req_dtype).npu()
    start_offset = torch.randint(
        0, ROW_LEN - step, (bs,), generator=gen, dtype=torch.int64
    ).npu()
    end_offset = start_offset + step
    golden = torch.cat(
        [
            req_to_token[int(r), int(s) : int(e)]
            for r, s, e in zip(req_pool_indices, start_offset, end_offset)
        ]
    )
    return req_to_token, req_pool_indices, start_offset, end_offset, golden


def run(bs, step, req_dtype, at_block_end, seed=0):
    pool, req, start, end, golden = make_case(bs, step, req_dtype, seed)
    n = bs * step
    if at_block_end:
        # >= 10 MB and a multiple of 2 MB, so the caching allocator hands this
        # allocation a block of its own; the view sits at its very end.
        backing = torch.empty((12 << 20) // 4, dtype=torch.int32, device=req.device)
        out = backing[backing.numel() - n :]
    else:
        out = torch.empty((n,), dtype=torch.int32, device=req.device)

    torch.ops.npu.cache_loc_update(req, pool, start, end, out)
    torch.npu.synchronize()

    assert torch.equal(
        out.cpu(), golden.cpu()
    ), f"cache_loc_update mismatch: bs={bs} step={step} dtype={req_dtype}"
    print(
        f"  bs={bs:4d} step={step:2d} {str(req_dtype).split('.')[-1]:>5s} "
        f"out={n:5d} int32 (kernel must not touch {bs * MAX_STEP:5d}) "
        f"{'at block end' if at_block_end else 'fresh block'}: ok"
    )


if __name__ == "__main__":
    # step < MAX_STEP is the overrun regime: the kernel would read and store back
    # bs * MAX_STEP entries for a tensor holding bs * step.
    print("out_cache_loc pinned at the end of its own block:")
    for bs, step in [(1, 4), (9, 4), (15, 4), (16, 4), (64, 1), (300, 2)]:
        run(bs, step, torch.int64, at_block_end=True)
    print("out_cache_loc in a fresh block, both index dtypes:")
    for req_dtype in (torch.int64, torch.int32):
        for bs, step in [(15, 4), (37, 3), (128, MAX_STEP)]:
            run(bs, step, req_dtype, at_block_end=False)
    print("PASSED")
