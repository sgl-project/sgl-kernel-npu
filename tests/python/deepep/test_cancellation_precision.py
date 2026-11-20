import argparse
import os
import random

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from utils import init_dist


def test(
    aligned_num_tokens: int,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    seed: int,
    local_rank: int,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]

    topk_weights_pos = (
        torch.randn((num_tokens, num_topk), dtype=torch.float32, device="npu").abs()
        + 0.1
    )
    sign_pattern = torch.randint(0, 2, (num_tokens, num_topk), device="npu")
    topk_weights = topk_weights_pos * torch.where(
        sign_pattern == 0, 1.0, -1.0
    ).float()

    masked_weights = topk_weights.masked_fill(topk_idx == -1, 0.0)
    weight_sums = masked_weights.sum(dim=1)
    near_zero_mask = weight_sums.abs() < 0.05
    near_zero_count = near_zero_mask.sum().item()

    packed_recv_x, _, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        aligned_num_tokens,
        num_experts,
        round_scale=False,
        async_finish=True,
        topk_weights=topk_weights,
        use_fp8=False,
    )

    simulated_gemm_x = packed_recv_x

    combined_x, _, _ = buffer.low_latency_combine(
        simulated_gemm_x,
        topk_idx,
        topk_weights,
        handle,
        async_finish=True,
        zero_copy=False,
        use_mxfp8=False,
    )

    golden_fp32 = x.float() * masked_weights.sum(dim=1, keepdim=True)
    golden_bf16 = golden_fp32.to(torch.bfloat16)

    abs_diff = torch.abs(combined_x.float() - golden_fp32)
    golden_abs = torch.abs(golden_fp32)

    rel_threshold = 0.1
    rel_mask = golden_abs > rel_threshold
    if rel_mask.any():
        max_rel_diff = (abs_diff[rel_mask] / golden_abs[rel_mask]).max().item()
    else:
        max_rel_diff = 0.0

    near_zero_golden_mask = golden_abs <= rel_threshold
    if near_zero_golden_mask.any():
        max_near_zero_abs = abs_diff[near_zero_golden_mask].max().item()
    else:
        max_near_zero_abs = 0.0

    max_abs_diff = abs_diff.max().item()

    print(
        f"[rank {rank}] max_rel_diff={max_rel_diff:.6f}, max_abs_diff={max_abs_diff:.6f}, "
        f"max_near_zero_abs={max_near_zero_abs:.6f}, "
        f"near_zero_tokens={near_zero_count}/{num_tokens}",
        flush=True,
    )

    assert not torch.isnan(combined_x).any().item(), "NaN detected in combined output"
    assert max_rel_diff < 0.05, (
        f"max_rel_diff={max_rel_diff:.6f} exceeds 5% threshold. "
        f"This indicates BF16 per-path rounding loss in async combine path."
    )
    assert max_near_zero_abs < 0.5, (
        f"max_near_zero_abs={max_near_zero_abs:.6f} exceeds 0.5 threshold. "
        f"Cancellation residual was lost due to early BF16 rounding."
    )

    if local_rank == 0:
        print(
            f" PASSED: {near_zero_count}/{num_tokens} near-cancellation tokens, "
            f"max_rel_diff={max_rel_diff:.6f}, max_near_zero_abs={max_near_zero_abs:.6f}",
            flush=True,
        )


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    base_num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    raw_num_tokens = base_num_tokens
    local_tokens_tensor = torch.tensor(
        [raw_num_tokens], dtype=torch.int32, device="npu"
    )
    dist.all_reduce(local_tokens_tensor, op=dist.ReduceOp.MAX)
    aligned_num_tokens = local_tokens_tensor.item()

    print(
        f"[rank {rank}] raw_num_tokens: {raw_num_tokens}, aligned_num_tokens: {aligned_num_tokens}"
    )

    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        aligned_num_tokens, hidden, num_ranks, num_experts
    )
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // num_ranks if num_ranks > 0 else 1,
        low_latency_strategy="default",
    )

    for seed in range(args.num_seeds):
        if rank == 0:
            print(f"\n=== Seed {seed} ===", flush=True)
        test(
            aligned_num_tokens,
            raw_num_tokens,
            hidden,
            num_experts,
            num_topk,
            rank,
            num_ranks,
            group,
            buffer,
            seed=seed,
            local_rank=local_rank,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test BF16 cancellation precision in low-latency combine"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=256, help="Number of tokens (default: 256)"
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
        "--num-seeds",
        type=int,
        default=3,
        help="Number of random seeds to test (default: 3)",
    )
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
