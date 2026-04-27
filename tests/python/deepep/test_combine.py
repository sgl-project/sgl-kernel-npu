import argparse
import os
import random

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from utils import calc_diff, init_dist, per_token_cast_back


def generate_base_inputs(args: argparse.Namespace, rank: int, num_ranks: int):
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts

    assert num_experts % num_ranks == 0
    experts_per_rank = num_experts // num_ranks
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    return {
        "num_tokens": num_tokens,
        "hidden": hidden,
        "num_topk": num_topk,
        "num_experts": num_experts,
        "experts_per_rank": experts_per_rank,
        "topk_idx": topk_idx,
    }


def generate_normal_combine_inputs(args: argparse.Namespace, rank: int, num_ranks: int):
    base_inputs = generate_base_inputs(args, rank, num_ranks)

    topk_weights = (
        torch.ones(
            (base_inputs["num_tokens"], base_inputs["num_topk"]),
            dtype=torch.float32,
            device="npu",
        )
        * rank
    )

    x = (
        torch.ones(
            (base_inputs["num_tokens"], base_inputs["hidden"]),
            dtype=torch.bfloat16,
            device="npu",
        )
        * rank
    )

    aligned_num_tokens = base_inputs["num_tokens"]

    return {
        "x": x,
        "topk_idx": base_inputs["topk_idx"],
        "topk_weights": topk_weights,
        "aligned_num_tokens": aligned_num_tokens,
        "num_tokens": base_inputs["num_tokens"],
        "hidden": base_inputs["hidden"],
        "num_experts": base_inputs["num_experts"],
    }


def generate_low_latency_combine_inputs(
    args: argparse.Namespace, rank: int, num_ranks: int
):
    base_inputs = generate_base_inputs(args, rank, num_ranks)
    num_tokens = base_inputs["num_tokens"]
    hidden = base_inputs["hidden"]
    num_topk = base_inputs["num_topk"]

    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()

    # Offset rank by 128 to test negative values in bfloat16 tensor
    # Constraint: num_ranks must be < 385 (128 + 257) to keep values in valid range
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding bfloat16 precision limit)"
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
    local_tokens_tensor = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    dist.all_reduce(local_tokens_tensor, op=dist.ReduceOp.MAX)
    aligned_num_tokens = local_tokens_tensor.item()

    return {
        "x": x,
        "topk_idx": base_inputs["topk_idx"],
        "topk_weights": topk_weights,
        "aligned_num_tokens": aligned_num_tokens,
        "num_tokens": num_tokens,
        "hidden": hidden,
        "num_experts": base_inputs["num_experts"],
    }


def test_only_normal_combine(
    args: argparse.Namespace,
    rank: int,
    num_ranks: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    print(f"[Rank {rank}] Starting PURE combine test (normal)...", flush=True)

    inputs = generate_normal_combine_inputs(args, rank, num_ranks)
    config = deep_ep.Config(24, 8, 256)

    (
        ref_num_tokens_per_rank,
        send_token_idx,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(inputs["topk_idx"], inputs["num_experts"])

    dispatch_args = {
        "x": inputs["x"],
        "num_tokens_per_rank": ref_num_tokens_per_rank,
        "is_token_in_rank": ref_is_token_in_rank,
        "num_tokens_per_expert": ref_num_tokens_per_expert,
        "config": config,
        "topk_idx": inputs["topk_idx"],
        "topk_weights": inputs["topk_weights"],
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

    combine_args = {
        "x": recv_x,
        "handle": handle,
        "config": config,
        "async_finish": False,
        "topk_weights": handle[7],
    }
    combined_x, _, _ = buffer.combine(**combine_args)
    check_x = combined_x.float()

    diff = calc_diff(
        check_x,
        inputs["x"]
        * handle[7].masked_fill(inputs["topk_idx"] == -1, 0).sum(dim=1).view(-1, 1),
    )
    assert diff < 5e-5, f"Combine diff too large: {diff}"

    print(f"[Rank {rank}] Normal combine test PASSED (diff: {diff:.6f})")


def test_only_low_latency_combine(
    args: argparse.Namespace,
    rank: int,
    num_ranks: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
    inputs: dict,
):
    print(f"[Rank {rank}] Starting PURE combine test (low latency)...", flush=True)

    num_local_experts = inputs["num_experts"] // num_ranks
    cumulative_stats = torch.zeros((num_local_experts,), dtype=torch.int, device="npu")

    packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
        inputs["x"],
        inputs["topk_idx"],
        inputs["aligned_num_tokens"],
        inputs["num_experts"],
        use_fp8=False,
        round_scale=False,
        use_ue8m0=False,
        cumulative_local_expert_recv_stats=cumulative_stats,
        async_finish=False,
        return_recv_hook=False,
    )
    simulated_gemm_x = (
        per_token_cast_back(*packed_recv_x)
        if isinstance(packed_recv_x, tuple)
        else packed_recv_x
    )

    out = torch.empty(
        (inputs["aligned_num_tokens"], inputs["hidden"]),
        dtype=torch.bfloat16,
        device="npu",
    )
    combined_x, event, hook = buffer.low_latency_combine(
        simulated_gemm_x,
        inputs["topk_idx"],
        inputs["topk_weights"],
        handle,
        async_finish=False,
        zero_copy=False,
        return_recv_hook=False,
        out=out,
    )

    diff = calc_diff(
        inputs["x"]
        * inputs["topk_weights"]
        .masked_fill(inputs["topk_idx"] == -1, 0)
        .sum(dim=1)
        .view(-1, 1),
        combined_x,
    )
    assert torch.isnan(combined_x).sum().item() == 0
    assert diff < 1e-5, f"Low latency combine diff too large: {diff}"

    print(f"[Rank {rank}] Low latency combine test PASSED (diff: {diff:.6f})")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    torch.npu.set_device(local_rank)
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank)

    if args.test_type == "normal":
        buffer = deep_ep.Buffer(
            group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
        )
        test_only_normal_combine(args, rank, num_ranks, buffer, group)
    elif args.test_type == "low_latency":
        inputs = generate_low_latency_combine_inputs(args, rank, num_ranks)
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            inputs["aligned_num_tokens"],
            inputs["hidden"],
            num_ranks,
            inputs["num_experts"],
        )
        buffer = deep_ep.Buffer(
            group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=inputs["num_experts"] // num_ranks,
        )
        test_only_low_latency_combine(args, rank, num_ranks, buffer, group, inputs)

    dist.barrier()
    dist.destroy_process_group()
    if local_rank == 0:
        print("\nAll combine tests PASSED!")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Test EP Combine Functions")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of processes to spawn (default: 16)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=512, help="Number of tokens (default: 512)"
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
        "--test-type",
        type=str,
        default="normal",
        choices=["normal", "low_latency"],
        help="Test type: normal (combine) or low_latency (low_latency_combine)",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
