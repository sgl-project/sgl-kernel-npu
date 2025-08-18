import os
import random
import time
import torch
import torch_npu
import torch.distributed as dist

from deep_ep import Buffer
from functools import partial
from utils import bench, calc_diff, hash_tensor

def test(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
         rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: Buffer, seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='npu') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='npu').to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='npu').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='npu').abs()

    # Check dispatch correctness
    do_check = True
    return_recv_hook = False
    hash_value, num_times = 0, 0

    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='npu')
    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                    use_fp8=False, round_scale=False, use_ue8m0=False,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
    simulated_gemm_x = packed_recv_x.clone()
    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='npu')
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    for i in range(num_local_experts if do_check else 0):
        expert_id = rank * num_local_experts + i
        temp = num_tokens / num_local_experts
        recv_x = packed_recv_x[i : int((i + 1) * temp)]
        recv_count = packed_recv_count[i]
        if i == 0:
            recv_layout_range = handle[1][(i + 1) * num_ranks - 1]
        else:
            recv_layout_range = handle[1][(i + 1) * num_ranks - 1] - handle[1][i * num_ranks - 1]

        # Check expert indices
        int_mask = (2 ** 32) - 1
        num_valid_tokens = recv_count.item()
        assert num_valid_tokens == (recv_layout_range & int_mask).item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.item()'
        assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}'

        if num_valid_tokens == 0:
            continue
        # Check received data
        recv_x = recv_x[:num_valid_tokens]
        recv_x_amin = recv_x[:, :-128].amin(dim=-1)
        assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
        hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

    # Check combine correctness
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
    combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                         async_finish=not return_recv_hook, zero_copy=False,
                                                         return_recv_hook=return_recv_hook, out=out)

    if do_check:
        diff = calc_diff(x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
        assert torch.isnan(combined_x).sum().item() == 0
        assert diff < 1e-5, f'Error: {diff=}, {zero_copy=}'
        hash_value ^= hash_tensor(combined_x)

    # noinspection PyShadowingNames
    def test_func(zero_copy: bool, return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=False, async_finish=False, return_recv_hook=return_recv_hook)
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                             zero_copy=zero_copy, return_recv_hook=return_recv_hook)

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, zero_copy=False, return_recv_hook=False))
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)
    
    return hash_value

def test_main():
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '17621'))
    world_size = int(os.getenv('WORLD_SIZE', 16))
    rank = int(os.getenv('RANK', 0))
    shared_expert_rank_num = int(os.getenv('MOE_SHARED_EXPERT_RANK_NUM', 0))

    dist.init_process_group(
        backend="hccl",
        init_method=f'tcp://{ip}:{port}',
        world_size=world_size,
        rank=rank
    )
    torch.npu.set_device(rank)
    group = dist.new_group(list(range(world_size)))
    print("===========group", group.size())
    if shared_expert_rank_num == 0:
        num_tokens, hidden, num_topk, num_experts = 128, 7168, 8, 288
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world_size, num_experts)
        buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                        num_qps_per_rank=num_experts // world_size)

        use_experts = num_experts
        use_ranks = world_size
    else:
        num_tokens, hidden, num_topk, num_experts = 1, 7168, 8, 31
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world_size, num_experts)
        buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                        num_qps_per_rank=num_experts // world_size)

        use_experts = num_experts - 1
        use_ranks = world_size - shared_expert_rank_num

    do_pressure_test = False
    for seed in range(int(1e9) if do_pressure_test else 1):
        if rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        ref_hash = test(num_tokens, hidden, use_experts, num_topk, rank, use_ranks, group, buffer, seed)
        for i in range(20):
            assert test(num_tokens, hidden, use_experts, num_topk, rank, use_ranks, group, buffer, seed) == ref_hash, f'Error: seed={seed}'
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    test_main()
