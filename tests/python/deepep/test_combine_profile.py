"""Profile combine correctness test for cam_moe_combine_normal_a5."""

import argparse
import os
import tempfile
from typing import Optional

import deep_ep
import torch
import torch.distributed as dist
import torch_npu
from utils import calc_diff, init_dist, per_token_cast_back, profile_npu_event_sequences

torch_npu.npu.config.allow_internal_format = True

# Combine profile event patterns: one or more NPU events that should appear
# when the combine operator is profiled.
COMBINE_EVENT_PATTERNS = (("CamMoeCombineNormal", "aclnnCamMoeCombineNormal"),)


def test_combine_profile(
    args: argparse.Namespace,
    num_local_ranks: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    num_tokens = args.num_tokens
    hidden = args.hidden
    num_topk = args.num_topk
    num_experts = args.num_experts
    assert num_experts % num_ranks == 0

    # Create random topk_idx
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    # topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = torch.zeros((num_tokens, num_topk), dtype=torch.int64, device='npu')
    for t in range(num_tokens):
        start = (t * num_topk) % num_experts
        for k in range(num_topk):
            topk_idx[t, k] = (start + k) % num_experts
    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )

    x = torch.randn((num_tokens, hidden), device="npu").bfloat16()

    # Get dispatch layout
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    config = deep_ep.Buffer.get_combine_config(num_ranks)

    # Prepare combine inputs via a (non-profiled) dispatch.
    (
        recv_x,
        _,
        _,
        _,
        handle,
        _,
    ) = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=config,
    )
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
    topk_weights_recv = handle[7]

    def combine_fn(profile_enable: bool):
        return buffer.combine(
            x=recv_x,
            handle=handle,
            topk_weights=topk_weights_recv,
            config=config,
            async_finish=False,
            profile_enable=profile_enable,
        )

    # Step 1: Run without profiling to verify correctness
    (combined_x, _, _) = combine_fn(profile_enable=False)
    expected = (
        x * topk_weights_recv.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
    )
    diff = calc_diff(combined_x.float(), expected)
    assert diff < 5e-5, f"Combine diff too large: {diff}"
    if rank == 0:
        print(
            f"[rank{rank}] combine without profiling OK. diff={diff:.6f}, "
            f"recv_x shape={recv_x.shape}",
            flush=True,
        )

    torch.npu.synchronize()

    # Step 2: Run with kernel-level profiling (begin_profile/end_profile)
    if args.kernel_trace_dir is not None:
        kernel_trace_dir = args.kernel_trace_dir
        os.makedirs(kernel_trace_dir, exist_ok=True)

        if rank == 0:
            print(
                f"[rank{rank}] begin kernel profiling: warmups={args.num_warmups}, "
                f"tests={args.num_tests}, trace_dir={kernel_trace_dir}",
                flush=True,
            )

        buffer.runtime.begin_profile(
            args.num_warmups, args.num_tests, kernel_trace_dir
        )

        try:
            for _ in range(args.num_warmups):
                combine_fn(profile_enable=True)
                torch.npu.synchronize()

            for _ in range(args.num_tests):
                combine_fn(profile_enable=True)
                torch.npu.synchronize()
        finally:
            print(
                f"[rank{rank}] end kernel profiling: trace_dir={kernel_trace_dir}",
                flush=True,
            )
            buffer.runtime.end_profile()

    dist.barrier()
    if rank == 0:
        print("[all] combine profiling test PASSED", flush=True)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)
    torch.manual_seed(rank)

    test_combine_profile(
        args, num_local_ranks, local_rank, num_ranks, rank, buffer, group
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test combine profiling for cam_moe_combine_normal_a5"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to spawn (default: 8)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4096,
        help="Number of tokens (default: 4096)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=7168,
        help="Hidden dimension size (default: 7168)",
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=8,
        help="Number of top-k experts (default: 8)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=256,
        help="Number of experts (default: 256)",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=2,
        help="Number of profiling warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of profiling test iterations (default: 5)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Optional directory to export host-level profiler chrome traces.",
    )
    parser.add_argument(
        "--kernel-trace-dir",
        type=str,
        default=None,
        help="Optional directory to export kernel-level profiler traces.",
    )
    parser.add_argument(
        "--dump-profile-events",
        action="store_true",
        help="Print matched profiler iterations and discovered event names for debugging.",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
