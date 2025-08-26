import torch
import torch_npu
import time

import sgl_kernel_npu


def assign_req_to_token_pool_native(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    bs: int,
):
    device = req_to_token.device

    tensor_zero = torch.tensor([0], device=device, dtype=torch.int32)
    bs_range = torch.arange(bs, device=device, dtype=torch.int32)
    lengths = end_offset - start_offset
    total_length = lengths.sum()
    cumsum_lengths = torch.cat(
        [tensor_zero, torch.cumsum(lengths, dim=0)[:-1].to(dtype=torch.int32)]
    )

    row_indices = torch.repeat_interleave(bs_range, lengths)
    repeat_cumsum = torch.repeat_interleave(cumsum_lengths, lengths)
    arange_total = torch.arange(total_length, device=device, dtype=torch.int32)
    offsets = arange_total - repeat_cumsum

    col_indices = (
        torch.repeat_interleave(start_offset.to(dtype=torch.int32), lengths) + offsets
    )
    src_indices = torch.repeat_interleave(cumsum_lengths, lengths) + offsets

    token_pool = req_to_token[req_pool_indices]
    token_pool[row_indices, col_indices] = out_cache_loc[src_indices]
    req_to_token[req_pool_indices] = token_pool


def assign_req_to_token_pool_golden(
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        start_offset: torch.Tensor,
        end_offset: torch.Tensor,
        out_cache_loc: torch.Tensor,
        bs: int,
):
    out_cache_loc_length = end_offset - start_offset
    token_pool = req_to_token[req_pool_indices]
    out_cache_loc_cumsum_length = torch.cumsum(out_cache_loc_length, dim=0)
    out_cache_loc_start_idx = torch.cat((torch.tensor([0], device=req_to_token.device), out_cache_loc_cumsum_length))
    
    for i in range(bs):
        token_pool[i][start_offset[i]:end_offset[i]] = out_cache_loc[
                                                       out_cache_loc_start_idx[i]:out_cache_loc_cumsum_length[i]]
    req_to_token[req_pool_indices] = token_pool


def test_op(req_indx_type):
    golden_spend_time = 0
    torch_spend_time = 0
    ascendC_spend_time = 0
    start = 0
    iter = 20

    for i in range(iter):
        if i > 0:
            start = time.time()
        assign_req_to_token_pool_golden(
            req_pool_indices, token_pool_copy, start_offset, end_offset, out_cache_loc, bs)
    torch.npu.synchronize()
    golden_spend_time += (time.time() - start) * 1000

    for j in range(iter):
        if j > 0:
            start = time.time()
        assign_req_to_token_pool_native(req_pool_indices, token_pool_copy2, start_offset, end_offset, out_cache_loc, bs)
    torch.npu.synchronize()
    torch_spend_time += (time.time() - start) * 1000

    for k in range(iter):
        if k > 0:
            start = time.time()
        torch.ops.npu.cache_loc_assign(req_pool_indices, token_pool, start_offset, end_offset, out_cache_loc)
    torch.npu.synchronize()
    ascendC_spend_time += (time.time() - start) * 1000

    accuracy = (token_pool == token_pool_copy).all() and (token_pool == token_pool_copy2).all()
    diff_num = ((token_pool != token_pool_copy).sum()).cpu()
    assert accuracy == True
    assert diff_num == torch.tensor([0])
    print(f"{req_indx_type} accuracy: {accuracy}")
    print(f"{req_indx_type} diff_num: {diff_num}")
    print(f"{req_indx_type} golden spend time:  {golden_spend_time / iter} ms")
    print(f"{req_indx_type} torch native spend time:  {torch_spend_time / iter} ms")
    print(f"{req_indx_type} ascend C spend time:  {ascendC_spend_time / iter} ms")


if __name__ == '__main__':
    bs = 50
    max_seq_len = 16384
    max_cache_loc = 10000

    token_pool = torch.arange(0, max_seq_len, device='npu', dtype=torch.int32)
    token_pool = token_pool.repeat(2000, 1)
    token_pool_copy = token_pool.clone()
    token_pool_copy2 = token_pool.clone()
    start_offset = torch.randint(2, max_seq_len, (bs,), device='npu', dtype=torch.int64) - 2
    end_offset = start_offset + torch.randint(1, 3, (bs,), device='npu', dtype=torch.int64)

    out_cache_loc_length = end_offset - start_offset
    out_cache_loc_cumsum_length = torch.cumsum(out_cache_loc_length, dim=0, dtype=torch.int32)
    out_cache_loc = torch.randint(0, max_cache_loc, (out_cache_loc_cumsum_length[-1],), device='npu', dtype=torch.int32)
    out_cache_loc_idx = torch.cat(
        (torch.tensor([0], device=token_pool.device, dtype=torch.int32), out_cache_loc_cumsum_length))

    # combo1: int64, int32, int64, int64, int32
    req_pool_indices = torch.arange(0, bs, device='npu', dtype=torch.int64)
    test_op("Int64")
    # combo2: int32, int32, int64, int64, int32
    req_pool_indices = torch.arange(0, bs, device='npu', dtype=torch.int32)
    test_op("Int32")
