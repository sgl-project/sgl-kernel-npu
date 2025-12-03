import argparse

import deep_ep
import torch
import torch.distributed as dist
from test_common import normal_test
from utils import calc_diff, init_dist, per_token_cast_back

RANK_OFFSET = 128


def low_latency_test(
    num_tokens: int,
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

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )

    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()

    return_recv_hook = False
    cumulative_local_expert_recv_stats = torch.zeros(
        (experts_per_rank,), dtype=torch.int, device="npu"
    )
    dispatch_use_fp8 = True
    packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
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

    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
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

    diff = calc_diff(
        x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
        combined_x,
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
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=True, num_qps_per_rank=1
    )

    normal_num_tokens = args.normal_num_tokens
    print("Start executing normal test...", flush=True)
    normal_test(
        normal_num_tokens,
        hidden,
        num_experts,
        num_topk,
        buffer,
    )
    print("End executing normal test...", flush=True)
    dist.barrier()

    low_latency_num_tokens = args.low_latency_num_tokens
    print("Start executing low latency test...", flush=True)
    low_latency_test(
        low_latency_num_tokens,
        hidden,
        num_experts,
        num_topk,
        rank,
        num_ranks,
        buffer,
    )
    print("End executing low latency test...", flush=True)
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

    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
