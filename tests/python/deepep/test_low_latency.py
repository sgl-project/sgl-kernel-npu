import os
import random
import time
import torch
import torch_npu
import torch.distributed as dist

from deep_ep import Buffer

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

    # Check combine correctness
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
    combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                         async_finish=not return_recv_hook, zero_copy=False,
                                                         return_recv_hook=return_recv_hook, out=out)
    # Check fused_deep_moe correctness
    # output, event, hook = buffer.fused_deep_moe()
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
        num_tokens, hidden, num_topk, num_experts = 1, 7168, 8, 16
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world_size, num_experts)
        buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                        num_qps_per_rank=num_experts // world_size)

        test(num_tokens, hidden, num_experts, num_topk, rank, world_size, group, buffer, seed=1)
    else:
        num_tokens, hidden, num_topk, num_experts = 1, 7168, 8, 31
        num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world_size, num_experts)
        buffer = Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                        num_qps_per_rank=num_experts // world_size)

        test(num_tokens, hidden, num_experts - 1, num_topk, rank, world_size - shared_expert_rank_num,
             group, buffer, seed=1)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    test_main()
