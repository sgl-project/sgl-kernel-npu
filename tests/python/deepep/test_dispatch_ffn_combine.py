import argparse
import os
import random
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from utils import bench_kineto, calc_diff, hash_tensor, init_dist

torch_npu.npu.config.allow_internal_format = True


# ======================== Main Test ========================
def test(
    num_tokens: int,
    hidden: int,
    moe_intermediate_size: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    local_rank: int,
    use_fp8: int = 1,
    seed: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    m = num_tokens
    k = hidden
    n = moe_intermediate_size
    topk = num_topk
    e = num_local_experts
    k2 = n // 2
    n2 = k

    num_tokens_tensor = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    dist.all_reduce(num_tokens_tensor, op=dist.ReduceOp.MAX)
    max_num_tokens = num_tokens_tensor.item()

    if local_rank == 0:
        print(
            f"[config] {num_tokens=}, {hidden=}, {moe_intermediate_size=}, {num_topk=}, {num_experts=}, {max_num_tokens=}",
            flush=True,
        )

    def generate_random_tensor(size, dtype):
        if dtype in [torch.float16, torch.bfloat16, torch.float32]:
            return torch.randn(size=size, dtype=dtype)
        elif dtype is torch.int8:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        elif dtype is torch.int32:
            return torch.randint(-1024, 1024, size=size, dtype=dtype)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

    expert_idx = torch.randint(0, num_experts, (m, topk), dtype=torch.int32).npu()
    # expert_idx = torch.zeros((m, topk), dtype=torch.int32, device='npu')
    # for t in range(m):
    #     start = (t * topk) % num_experts
    #     for j in range(topk):
    #         expert_idx[t, j] = (start + j) % num_experts
    weight_dtype = torch.int8 if use_fp8 else torch.bfloat16

    def run_normal():
        x = generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        weight1 = generate_random_tensor((e, k, n), dtype=weight_dtype).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = generate_random_tensor((e, k2, n2), dtype=weight_dtype).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64).npu()
        scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64).npu()

        probs = torch.randn(size=(m, topk), dtype=torch.float32).npu()

        out, expert_token_nums = buffer.fused_deep_moe(
            x=x,
            topk_idx=expert_idx,
            topk_weights=probs,
            gmm1_permuted_weight=weight1,
            gmm1_permuted_weight_scale=scale1,
            gmm2_weight=weight2,
            gmm2_weight_scale=scale2,
            num_max_dispatch_tokens_per_rank=(
                max_num_tokens * topk * 2
            ),  # m * topK * 2
            num_experts=num_experts,
            quant_mode=1,
            fuse_mode=2,
        )
        # print(f"[run_normal] {rank=}, {x[:2, :10]=}, {scale1[:2, :10]=}, {scale2[:2, :10]=}, {out[:2, :10]=}", flush=True)
        return out, expert_token_nums

    out_1, expert_token_nums_1 = run_normal()
    print(f"{rank=}, {out_1.shape=}, {expert_token_nums_1=}", flush=True)
    print(f"{rank=} run_normal End")

    if local_rank == 0:
        print(f"{rank=} PASSED")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = Buffer(group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1)
    print(f"[Rank {rank}] Buffer created OK.", flush=True)
    torch.manual_seed(rank)

    num_tokens, hidden, moe_intermediate_size = (
        args.num_tokens,
        args.hidden,
        args.moe_intermediate_size,
    )
    num_topk, num_experts = args.num_topk, args.num_experts
    use_fp8 = args.use_fp8

    for i in range(5):
        test(
            num_tokens,  # m
            hidden,  # k --> n2
            moe_intermediate_size,  # n --> k2
            num_experts,  # num_local_experts --> e
            num_topk,  # topk
            rank,
            num_ranks,
            group,
            buffer,
            num_local_ranks,
            use_fp8,
            seed=1,
        )
        if local_rank == 0:
            print(f"loop {i=} finish.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dispatch_ffn_combine kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=256, help="Number of tokens (default: 256)"
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=4096,
        help="Moe intermediate size (default: 4096)",
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--use-fp8", type=int, default=1, help="Number of use-fp8 (default: 1)"
    )

    args = parser.parse_args()
    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
