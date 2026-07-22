import argparse
import os
import random
import time
from typing import Optional

# noinspection PyUnresolvedReferences
import deep_ep
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from utils import (
    bench,
    calc_diff,
    diagnose_matrix,
    init_dist,
    inplace_unique,
    per_token_cast_back,
)

# 设置环境变量
env = os.environ.copy()
env["DEEP_USE_MODE"] = "alltoall"


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    num_local_ranks: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    base_num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    enable_dynamic_tokens = args.enable_dynamic_tokens
    quant_type = args.quant_type  # no, int8
    if quant_type == "no":
        quant_type_tensor = None
    elif quant_type == "int8":
        quant_type_tensor = torch.tensor([], dtype=torch.int8, device="npu")

    num_servers = num_ranks // num_local_ranks
    expert_token_nums_type = int(os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1))

    if enable_dynamic_tokens:
        fluctuation_percentage = 0.1
        min_fluctuation = 2

        if base_num_tokens < 10:
            fluctuation = random.randint(-min_fluctuation, min_fluctuation)
            num_tokens = base_num_tokens + fluctuation
        else:
            fluctuation = random.uniform(
                1 - fluctuation_percentage, 1 + fluctuation_percentage
            )
            num_tokens = int(base_num_tokens * fluctuation)

        # Ensure num_tokens is at least 1
        num_tokens = max(num_tokens, 1)
    else:
        num_tokens = base_num_tokens

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, active_ranks={args.active_ranks}",
            flush=True,
        )

    experts_per_rank = num_experts // num_ranks

    if args.active_ranks:
        # Only assign tokens to the specified ranks
        try:
            active_ranks = [
                int(r.strip()) for r in args.active_ranks.split(",") if r.strip()
            ]
        except ValueError:
            raise ValueError(
                f"Invalid value in --active-ranks: {args.active_ranks}. "
                f"Must be a comma-separated list of integers, e.g., '0,1,3'."
            )

        # Validate range
        if any(r < 0 or r >= num_ranks for r in active_ranks):
            raise ValueError(
                f"Invalid rank in --active-ranks: {active_ranks}. "
                f"Ranks must be in range [0, {num_ranks-1}]."
            )

        if not active_ranks:
            raise ValueError(
                "Parsed --active-ranks is empty. Provide at least one valid rank."
            )

        valid_experts = torch.cat(
            [
                torch.arange(
                    r * experts_per_rank, (r + 1) * experts_per_rank, device="npu"
                )
                for r in active_ranks
            ]
        )
        # Randomly sample experts from active ranks only
        topk_idx = valid_experts[
            torch.randint(0, len(valid_experts), (num_tokens, num_topk), device="npu")
        ]
    else:
        # Default: random over all experts (original behavior)
        scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="npu"
            ).abs()
            + 1
        )
        topk_idx = torch.zeros((num_tokens, num_topk), dtype=torch.int64, device='npu')
        for t in range(num_tokens):
            start = (t * num_topk) % num_experts
            for k in range(num_topk):
                topk_idx[t, k] = (start + k) % num_experts
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="npu") * rank
    )
    rank_idx = topk_idx // experts_per_rank
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts), dtype=torch.int, device="npu")
    for expert_id in range(num_experts):
        num_tokens_per_expert[expert_id] = (topk_idx == expert_id).sum()

    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, op=dist.ReduceOp.SUM, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="npu")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="npu"
    )

    for i in range(num_ranks):
        token_sel = (rank_idx == i).max(dim=-1)[0]  # [num_tokens]
        token_indices = torch.nonzero(token_sel, as_tuple=True)[0]  # [count]
        count = token_indices.numel()
        num_tokens_per_rank[i] = count
        if count > 0:
            token_idx_in_rank[i][token_indices] = torch.arange(count, device="npu")

    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = (token_idx_in_rank >= 0).to(torch.bool)
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    print(f"[layout] {rank=} Kernel performance: {t * 1000:.3f} ms", flush=True)
    print("", flush=True)
    dist.barrier()
    time.sleep(1)

    return_values = buffer.get_dispatch_layout(topk_idx, num_experts)
    (
        ref_num_tokens_per_rank,
        _,
        ref_num_tokens_per_expert,
        ref_is_token_in_rank,
        _,
    ) = return_values
    print(f"rank {rank} layout PASSED")

    # Config
    buffer_size = 256
    config = deep_ep.Config(24, 8, buffer_size)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float32, device="npu") * rank
    )
    topk_weights_pure_rand = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    )

    # Test dispatch
    for current_x in filter(lambda elem: elem is not None, (x, x_pure_rand)):
        if local_rank == 0:
            print(
                f'[testing] Running with {"INT8" if isinstance(current_x, tuple) else "BF16"}, with top-k {num_topk} ...',
                flush=True,
            )
        dispatch_args = {
            "x": (
                current_x
                if quant_type_tensor is None
                else (current_x, quant_type_tensor)
            ),
            "num_tokens_per_rank": ref_num_tokens_per_rank,
            "is_token_in_rank": ref_is_token_in_rank,
            "num_tokens_per_expert": ref_num_tokens_per_expert,
            "config": config,
            "topk_idx": topk_idx,
            "topk_weights": (
                topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
            ),
        }

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(**dispatch_args)
        recv_x_original = (
            recv_x[0].view(torch.uint8).clone().view(recv_x[0].dtype)
            if isinstance(recv_x, tuple)
            else recv_x.clone()
        )
        quant_scales = (
            recv_x[1].view(torch.uint8).clone().view(recv_x[1].dtype)
            if isinstance(recv_x, tuple)
            else None
        )
        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

        # Checks
        local_expert_token = gbl_num_tokens_per_expert.view(num_ranks, -1)[rank]
        if expert_token_nums_type == 0:
            local_expert_token_list = local_expert_token.cumsum(
                dim=0
            ).tolist()  # 计算前缀和并转为 list
        else:
            local_expert_token_list = local_expert_token.tolist()

        assert (
            local_expert_token_list == recv_num_tokens_per_expert_list
        ), f"Assertion num_tokens_per_rank failed on rank {rank}: Expected {local_expert_token_list}, Actual {recv_num_tokens_per_expert_list}"

        # Test combine
        combine_args = {
            "x": recv_x,
            "handle": handle,
            "config": config,
            "async_finish": False,
            "topk_weights": handle["topk_weights"],
        }
        combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
        check_x = combined_x.float()

        ref_x = x_pure_rand if current_x is x_pure_rand else x
        diff = calc_diff(
            check_x,
            ref_x * handle["topk_weights"].masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
        )
        golden = ref_x * handle["topk_weights"].masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
        # translate all zeros to eps in golden
        eps = 1e-8
        golden_nozero = torch.where(golden == 0, eps, golden)
        max_diff = torch.max(torch.abs(check_x - golden) / golden_nozero).item()

        avg_diff = torch.mean(torch.abs(check_x - golden) / golden_nozero).item()
        print(f"{rank=}, {avg_diff=:.5f}, {max_diff=:.5f}, cosine_diff={diff:.5f}")
        assert (diff < 5e-5), f"Assertion diff failed on {rank=}"

        # For later tuning
        dispatch_bf16_recv_bytes = recv_x.numel() * 2
        combine_bf16_send_bytes = dispatch_bf16_recv_bytes

        if local_rank == 0:
            print(f"[test] passed, {dispatch_bf16_recv_bytes=}", flush=True)
    if local_rank == 0:
        print("", flush=True)


    # Tune dispatch performance
    def calculate_recv_bytes(dispatch_bf16_recv_bytes, quant_type):
        hidden_dim = hidden
        bs = dispatch_bf16_recv_bytes / 2 / hidden_dim
        num_values = bs * hidden_dim
        
        if quant_type == "no":
            # No quantization, use original BF16 communication
            recv_bytes = dispatch_bf16_recv_bytes
        elif quant_type == "int8":
            # INT8 per-token quantization:
            # - Data: num_values * 1 byte (INT8)
            # - Scale: x tokens * 2 bytes each (BF16)
            data_bytes = num_values * 1
            scale_bytes = bs * 2
            recv_bytes = data_bytes + scale_bytes
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type}")
        return recv_bytes


    config = deep_ep.Config(24, 8, buffer_size)
    for current_x in filter(lambda elem: elem is not None, (x,)):
        # Replace the original recv_bytes calculation with:
        recv_bytes = calculate_recv_bytes(dispatch_bf16_recv_bytes, quant_type)

        tune_args = {
            "x": (
                current_x
                if quant_type_tensor is None
                else (current_x, quant_type_tensor)
            ),
            "config": config,
            "num_tokens_per_rank": ref_num_tokens_per_rank,
            "is_token_in_rank": ref_is_token_in_rank,
            "num_tokens_per_expert": ref_num_tokens_per_expert,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
        }

        t = bench(lambda: buffer.dispatch(**tune_args))[0]
        if local_rank == 0:
            print(
                f'[tuning] Dispatch ({quant_type=}) {recv_bytes / 1e9 / t:.2f} GB/s (HCCS), avg_t: {t * 1e6:.2f} us',
                flush=True,
            )
            print("", flush=True)

    dispatch_args = {
        "x": x if quant_type_tensor is None else (x, quant_type_tensor),
        "num_tokens_per_rank": ref_num_tokens_per_rank,
        "is_token_in_rank": ref_is_token_in_rank,
        "num_tokens_per_expert": ref_num_tokens_per_expert,
        "config": config,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
    # Tune combine performance
    tune_args = {
        "x": recv_x,
        "handle": handle,
        "config": config,
        "async_finish": False,
        "topk_weights": handle["topk_weights"],
    }
    t = bench(lambda: buffer.combine(**tune_args))[0]
    if local_rank == 0:
        print(
            f"[tuning] Combine {combine_bf16_send_bytes / 1e9 / t:.2f} GB/s (HCCS), avg_t: {t * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1,
        normal_strategy="alltoall",
        low_latency_strategy="alltoall",
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)
    torch.manual_seed(rank)

    test_main(args, num_local_ranks, local_rank, num_ranks, rank, buffer, group)
    if local_rank == 0:
        print("", flush=True)

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
        "--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)"
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
        "--active-ranks",
        type=str,
        default="",
        help="Comma-separated list of ranks that will receive tokens. "
        'Example: "0,1,3". If empty, all ranks may receive tokens.',
    )
    parser.add_argument(
        "--enable-dynamic-tokens",
        action="store_true",
        help="Whether to enable dynamic tokens for testing",
    )
    parser.add_argument(
        "--quant-type",
        dest="quant_type",
        type=str,
        default="no",
        help="quant type: no, int8",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
