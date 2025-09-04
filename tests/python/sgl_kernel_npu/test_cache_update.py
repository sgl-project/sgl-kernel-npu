import time

import sgl_kernel_npu
import torch
import torch_npu


def assign_extend_cache_locs_native(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    bs,
):
    out_cache_loc_length = end_offset - start_offset
    token_pool = req_to_token[req_pool_indices]
    out_cache_loc_cumsum_length = torch.cumsum(out_cache_loc_length, dim=0)
    out_cache_loc_cumsum_length = torch.cat(
        (
            torch.tensor([0], device=out_cache_loc_length.device),
            out_cache_loc_cumsum_length,
        )
    )
    for i in range(bs):
        out_cache_loc[
            out_cache_loc_cumsum_length[i] : out_cache_loc_cumsum_length[i]
            + out_cache_loc_length[i]
        ] = token_pool[i][start_offset[i] : end_offset[i]]
    return out_cache_loc


def test_op(req_indx_type):
    torch.npu.synchronize()

    golden_spend_time = 0
    ascendC_spend_time = 0
    start = 0
    iter = 20
    for i in range(iter):
        if i == 1:
            start = time.time()
        assign_extend_cache_locs_native(
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, bs
        )
    torch.npu.synchronize()
    golden_spend_time += (time.time() - start) * 1000
    print(f"golden_spend_time: {golden_spend_time / iter} ms")

    for j in range(iter):
        if j == 1:
            start = time.time()
        torch.ops.npu.cache_loc_update(
            req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc_copy
        )

    torch.npu.synchronize()
    ascendC_spend_time += (time.time() - start) * 1000
    accuracy = (out_cache_loc == out_cache_loc_copy).all()
    diff_num = (out_cache_loc != out_cache_loc_copy).sum().cpu()
    print(f"{req_indx_type} ascendC_spend_time: {ascendC_spend_time / iter} ms")
    print(f"{req_indx_type} accuracy: {accuracy}")
    print(f"{req_indx_type} diff_num: {diff_num}")
    assert accuracy == True
    assert diff_num == torch.tensor([0])


if __name__ == "__main__":
    bs = 300
    max_seq_len = 8192
    max_cache_loc = 10000

    req_to_token = torch.arange(0, max_seq_len, device="npu", dtype=torch.int32)
    req_to_token = req_to_token.repeat(2000, 1)
    start_offset = (
        torch.randint(2, max_seq_len, (bs,), device="npu", dtype=torch.int64) - 2
    )
    end_offset = start_offset + torch.randint(
        1, 3, (bs,), device="npu", dtype=torch.int64
    )

    out_cache_loc_length = end_offset - start_offset
    out_cache_loc_cumsum_length = torch.cumsum(
        out_cache_loc_length, dim=0, dtype=torch.int32
    )
    out_cache_loc = torch.randint(
        0,
        max_cache_loc,
        (out_cache_loc_cumsum_length[-1],),
        device="npu",
        dtype=torch.int64,
    )
    out_cache_loc_copy = out_cache_loc.clone()
    out_cache_loc_idx = torch.cat(
        (
            torch.tensor([0], device=req_to_token.device, dtype=torch.int32),
            out_cache_loc_cumsum_length,
        )
    )

    out_cache_loc_copy = out_cache_loc_copy.to(torch.int32)
    # combo1: int64, int32, int64, int64, int32
    req_pool_indices = torch.arange(0, bs, device="npu", dtype=torch.int64)
    test_op("int64")

    # combo1: int32, int32, int64, int64, int32
    req_pool_indices = torch.arange(0, bs, device="npu", dtype=torch.int32)
    test_op("int32")
