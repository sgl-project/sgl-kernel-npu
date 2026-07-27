"""EP16 smoke test for the MiniMax-M3 normal-mode FuseEP operator.

The second projection is zeroed so the final routed output has an exact zero
reference while GMM1 still executes a nonzero SwiGLU-OAI activation path.
"""

import argparse

import torch
import torch.distributed as dist
import torch_npu

from deep_ep import Buffer
from utils import init_dist


def float_to_int64_bits(scale: torch.Tensor) -> torch.Tensor:
    return scale.contiguous().view(torch.int32).to(torch.int64)


def make_weights(num_local_experts: int):
    # w1 is nonzero so the SwiGLU-OAI epilogue is exercised. A zero w2 gives
    # an exact final-output reference without reproducing distributed routing.
    w1 = torch.ones(
        (num_local_experts, 6144, 6144), dtype=torch.int8, device="npu"
    )
    w2 = torch.zeros(
        (num_local_experts, 3072, 6144), dtype=torch.int8, device="npu"
    )
    torch_npu.npu_format_cast_(w1, 29)
    torch_npu.npu_format_cast_(w2, 29)
    scale1 = float_to_int64_bits(
        torch.ones((num_local_experts, 6144), dtype=torch.float32, device="npu")
    )
    scale2 = float_to_int64_bits(
        torch.ones((num_local_experts, 6144), dtype=torch.float32, device="npu")
    )
    return w1, scale1, w2, scale2


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    assert world_size == 16, "DispatchFFNCombineM3 is specialized for EP16"

    num_tokens = args.num_tokens
    num_experts = 128
    topk = 4
    experts_per_rank = num_experts // world_size
    x = torch.ones((num_tokens, 6144), dtype=torch.bfloat16, device="npu")
    topk_ids = (
        torch.arange(num_tokens * topk, device="npu", dtype=torch.int32)
        .reshape(num_tokens, topk)
        % num_experts
    )
    topk_weights = torch.ones((num_tokens, topk), dtype=torch.float32, device="npu")
    w1, scale1, w2, scale2 = make_weights(experts_per_rank)

    buffer = Buffer(group, low_latency_mode=False)
    output, expert_token_nums = buffer.dispatch_ffn_combine_m3(
        x,
        topk_ids,
        topk_weights,
        w1,
        scale1,
        w2,
        scale2,
        # Worst-case routed tokens that can arrive at one EP rank.
        num_tokens * topk * world_size,
        num_experts,
    )

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    local_counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).to(torch.int32)
    dist.all_reduce(local_counts, group=group)
    expected_counts = local_counts[
        rank * experts_per_rank : (rank + 1) * experts_per_rank
    ]
    torch.testing.assert_close(expert_token_nums, expected_counts)

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=16)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        run, args=(args.num_processes, args), nprocs=args.num_processes
    )
