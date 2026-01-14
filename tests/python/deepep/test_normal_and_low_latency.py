import argparse
import random

import deep_ep
import torch
import torch.distributed as dist
from test_common import normal_test
from utils import calc_diff, init_dist, per_token_cast_back

RANK_OFFSET = 128


def low_latency_test(
    aligned_num_tokens: int,
    actual_num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    buffer: deep_ep.Buffer,
):
    experts_per_rank = num_experts // num_ranks

    rank_offset = RANK_OFFSET
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.zeros((aligned_num_tokens, hidden), dtype=torch.bfloat16, device="npu")

    if actual_num_tokens > 0:
        x[:actual_num_tokens] = torch.ones(
            (actual_num_tokens, hidden), dtype=torch.bfloat16, device="npu"
        ) * (rank - rank_offset)
        x[:actual_num_tokens, -128:] = (
            torch.arange(actual_num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
        )

    scores = (
        torch.randn(
            (aligned_num_tokens, num_experts), dtype=torch.float32, device="npu"
        ).abs()
        + 1
    )

    topk_idx = torch.full(
        (aligned_num_tokens, num_topk), -1, dtype=torch.long, device="npu"
    )

    if actual_num_tokens > 0:
        actual_scores = scores[:actual_num_tokens]
        actual_topk_idx = torch.topk(
            actual_scores, num_topk, dim=-1, largest=True, sorted=True
        )[1]
        topk_idx[:actual_num_tokens] = actual_topk_idx

    topk_weights = torch.zeros(
        (aligned_num_tokens, num_topk), dtype=torch.float32, device="npu"
    )
    if actual_num_tokens > 0:
        topk_weights[:actual_num_tokens] = torch.randn(
            (actual_num_tokens, num_topk), dtype=torch.float32, device="npu"
        ).abs()

    return_recv_hook = False
    cumulative_local_expert_recv_stats = torch.zeros(
        (experts_per_rank,), dtype=torch.int, device="npu"
    )
    dispatch_use_fp8 = True
    packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x,
        topk_idx,
        aligned_num_tokens,
        num_experts,
        use_fp8=dispatch_use_fp8,
        round_scale=False,
        use_ue8m0=False,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        async_finish=not return_recv_hook,
        return_recv_hook=return_recv_hook,
    )
    simulated_gemm_x = (
        per_token_cast_back(*packed_recv_x) if dispatch_use_fp8 else packed_recv_x
    )

    (
        _,
        _,
        _,
        hidden,
        _,
        _,
    ) = handle

    out = torch.empty((aligned_num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    combined_x, event, hook = buffer.low_latency_combine(
        simulated_gemm_x,
        topk_idx,
        topk_weights,
        handle,
        async_finish=not return_recv_hook,
        zero_copy=False,
        return_recv_hook=return_recv_hook,
        out=out,
    )

    if actual_num_tokens > 0:
        # 计算期望的输出（只考虑有效token）
        expected_x = torch.zeros(
            (aligned_num_tokens, hidden), dtype=torch.bfloat16, device="npu"
        )
        expected_x[:actual_num_tokens] = torch.ones(
            (actual_num_tokens, hidden), dtype=torch.bfloat16, device="npu"
        ) * (rank - rank_offset)
        expected_x[:actual_num_tokens, -128:] = (
            torch.arange(actual_num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
        )

        diff = calc_diff(
            expected_x[:actual_num_tokens]
            * topk_weights[:actual_num_tokens]
            .masked_fill(topk_idx[:actual_num_tokens] == -1, 0)
            .sum(dim=1)
            .view(-1, 1),
            combined_x[:actual_num_tokens],
        )
        assert torch.isnan(combined_x).sum().item() == 0
        if dispatch_use_fp8:
            assert diff < 1e-4, f"Error: {diff=}"
        else:
            assert diff < 1e-5, f"Error: {diff=}"


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_topk, num_experts, hidden = args.num_topk, args.num_experts, args.hidden
    assert num_experts % num_ranks == 0
    torch.manual_seed(rank)

    for i in range(args.test_loop):
        buffer = deep_ep.Buffer(
            group, int(2e9), 0, low_latency_mode=True, num_qps_per_rank=1
        )
        base_normal_num_tokens = args.normal_num_tokens
        fluctuation_percentage = 0.1
        min_fluctuation = 2

        if base_normal_num_tokens < 10:
            fluctuation = random.randint(-min_fluctuation, min_fluctuation)
            normal_num_tokens = base_normal_num_tokens + fluctuation
        else:
            fluctuation = random.uniform(
                1 - fluctuation_percentage, 1 + fluctuation_percentage
            )
            normal_num_tokens = int(base_normal_num_tokens * fluctuation)

        # Ensure normal_num_tokens is at least 1
        normal_num_tokens = max(normal_num_tokens, 1)

        if local_rank == 0:
            print(f"Start executing normal test loop {i} ...", flush=True)
        normal_test(
            normal_num_tokens,
            hidden,
            num_experts,
            num_topk,
            buffer,
        )
        if local_rank == 0:
            print(f"End executing normal test loop {i} ...", flush=True)

        base_low_latency_num_tokens = args.low_latency_num_tokens

        if base_low_latency_num_tokens < 10:
            fluctuation = random.randint(-min_fluctuation, min_fluctuation)
            low_latency_num_tokens = base_low_latency_num_tokens + fluctuation
        else:
            fluctuation = random.uniform(
                1 - fluctuation_percentage, 1 + fluctuation_percentage
            )
            low_latency_num_tokens = int(base_low_latency_num_tokens * fluctuation)

        # Ensure low_latency_num_tokens is at least 1
        low_latency_num_tokens = max(low_latency_num_tokens, 1)

        local_tokens_tensor = torch.tensor(
            [low_latency_num_tokens], dtype=torch.int32, device="npu"
        )
        dist.all_reduce(local_tokens_tensor, op=dist.ReduceOp.MAX)
        aligned_num_tokens = local_tokens_tensor.item()

        if local_rank == 0:
            print(f"Start executing low latency test loop {i} ...", flush=True)
        low_latency_test(
            aligned_num_tokens,
            low_latency_num_tokens,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            buffer,
        )
        if local_rank == 0:
            print(f"End executing low latency test loop {i} ...", flush=True)
        del buffer
        torch.npu.empty_cache()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test intranode EP kernels")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--normal-num-tokens",
        type=int,
        default=4096,
        help="Number of normal tokens (default: 4096)",
    )
    parser.add_argument(
        "--low-latency-num-tokens",
        type=int,
        default=256,
        help="Number of low latency tokens (default: 256)",
    )
    parser.add_argument(
        "--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)"
    )
    parser.add_argument(
        "--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--test-loop",
        type=int,
        default=1000,
        help="Number of test loop (default: 1000)",
    )

    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
