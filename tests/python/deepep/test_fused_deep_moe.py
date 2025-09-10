import argparse
import os
import random
import time
from functools import partial

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from utils import bench, hash_tensor, init_dist


def generate_random_tensor(shape, dtype, int_lower=-16, int_upper=+16):
    if dtype in [torch.int8, torch.int32, torch.int64]:
        return torch.randint(int_lower, int_upper, shape, dtype=dtype)
    return torch.rand(shape, dtype=dtype)

def test(
        num_tokens: int,
        hidden: int,
        num_experts: int,
        num_topk: int,
        rank: int,
        num_ranks: int,
        group: dist.ProcessGroup,
        buffer: Buffer,
        seed: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)


    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert (
            num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (
            rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
    scores = (
            torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
            + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()
    group_num = group.size()
    GMM1_OUTPUT_SIZE = 4096
    gmm1PermutedWeight = generate_random_tensor([group_num, hidden, GMM1_OUTPUT_SIZE], torch.int8).npu()   # ND
    gmm1PermutedWeightScale = generate_random_tensor([group_num, GMM1_OUTPUT_SIZE], torch.float).abs().npu()
    GMM2_INPUT_SIZE = GMM1_OUTPUT_SIZE // 2
    GMM2_OUTPUT_SIZE = hidden
    gmm2Weight = generate_random_tensor([group_num, GMM2_OUTPUT_SIZE, GMM2_INPUT_SIZE], torch.int8).npu()   # ND
    gmm2WeightScale = generate_random_tensor([group_num, GMM2_OUTPUT_SIZE], torch.float).abs().npu()

    print(f"num_tokens = {num_tokens}, hidden = {hidden}, num_experts = {num_experts}, num_topk = {num_topk}")

    output, event, hook = buffer.fused_deep_moe(x, topk_idx, topk_weights,
                                                gmm1PermutedWeight, gmm1PermutedWeightScale, gmm2Weight, gmm2WeightScale,
                                                num_tokens, num_experts, use_fp8=False)

    # Verify output
    do_check = True
    hash_value = 0

    assert output.shape == (num_tokens, hidden), f"Output shape mismatch. Expected: {(num_tokens, hidden)}, Got: {output.shape}"
    assert output.dtype == torch.bfloat16, f"Output dtype mismatch. Expected: torch.bfloat16, Got: {output.dtype}"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    assert not torch.isnan(output).any(), "Output contains NaN values"

    if do_check:
        hash_value ^= hash_tensor(output)

    num_bf16_bytes = hidden * 2
    num_moe_comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_moe_comm_bytes += num_bf16_bytes * num_selections

    moe_args = {'x': x, 'topk_idx': topk_idx, 'topk_weights': topk_weights,
                'gmm1PermutedWeight': gmm1PermutedWeight, 'gmm1PermutedWeightScale': gmm1PermutedWeightScale,
                'gmm2Weight': gmm2Weight, 'gmm2WeightScale': gmm2WeightScale,
                'num_max_dispatch_tokens_per_rank': num_tokens, 'num_experts': num_experts, 'use_fp8': False}
    avg_t, min_t, max_t = bench(lambda: buffer.fused_deep_moe(**moe_args))
    print(f'[rank {rank}] moe bandwidth: {(num_moe_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)
    return hash_value


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    use_experts = num_experts if shared_expert_rank_num == 0 else (num_experts - 1)
    use_ranks = num_ranks - shared_expert_rank_num
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=use_experts // use_ranks if use_ranks > 0 else 1,
    )

    test(
        num_tokens,
        hidden,
        use_experts,
        num_topk,
        rank,
        use_ranks,
        group,
        buffer,
        seed=1,
    )

    do_pressure_test = False
    for seed in range(int(1e9) if do_pressure_test else 0):
        if rank == 0:
            print(f"Testing with seed {seed} ...", flush=True)
        ref_hash = test(
            num_tokens,
            hidden,
            use_experts,
            num_topk,
            rank,
            use_ranks,
            group,
            buffer,
            seed=seed,
        )
        for i in range(20):
            assert (
                    test(
                        num_tokens,
                        hidden,
                        use_experts,
                        num_topk,
                        rank,
                        use_ranks,
                        group,
                        buffer,
                        seed=seed,
                    )
                    == ref_hash
            ), f"Error: seed={seed}"
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fused deep moe")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=1, help="Number of tokens (default: 1)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=16, help="Number of experts (default: 16)"
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )