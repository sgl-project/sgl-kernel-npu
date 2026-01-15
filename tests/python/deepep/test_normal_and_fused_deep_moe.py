import argparse

import deep_ep
import torch
import torch.distributed as dist
from test_common import normal_test
from test_fused_deep_moe import init_base_weights, init_fused_weights_int8
from utils import init_dist

RANK_OFFSET = 128


def fused_deep_moe_test(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    buffer: deep_ep.Buffer,
):
    num_local_experts = num_experts // num_ranks
    rank_offset = RANK_OFFSET
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.rand((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * 10 - 5
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()
    w13_weight, w13_weight_scale, w2_weight, w2_weight_scale = init_base_weights(
        num_local_experts=num_local_experts,
        hidden_in=hidden,
    )
    w13, w13s, w2, w2s = init_fused_weights_int8(
        w13_weight.clone().detach(),
        w13_weight_scale.clone().detach(),
        w2_weight.clone().detach(),
        w2_weight_scale.clone().detach(),
    )

    fused_output, _ = buffer.fused_deep_moe(
        x,
        topk_idx,
        topk_weights,
        w13,
        w13s,
        w2,
        w2s,
        num_tokens,
        num_experts,
        0,
    )


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

    fused_moe_num_tokens = args.fused_moe_num_tokens
    print("Start executing fused deep moe test...", flush=True)
    fused_deep_moe_test(
        fused_moe_num_tokens,
        hidden,
        num_experts,
        num_topk,
        num_ranks,
        buffer,
    )
    print("End executing fused deep moe test...", flush=True)
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
        "--fused-moe-num-tokens",
        type=int,
        default=256,
        help="Number of fused deep moe tokens (default: 256)",
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
