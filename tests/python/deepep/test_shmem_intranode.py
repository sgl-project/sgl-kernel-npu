import argparse
import os
import random
import time
from functools import partial
from typing import Optional

# noinspection PyUnresolvedReferences
import deep_ep
import numpy as np
import shmem as ash
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch_npu
from utils import (
    bench,
    bench_kineto,
    calc_diff,
    calculate_avg_stats,
    diagnose_matrix,
    init_dist,
    inplace_unique,
    per_token_cast_back,
)

g_ash_size = 2 * 1024 * 1024 * 1024
G_IP_PORT = "tcp://127.0.0.1:8688"


def register_shmem():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")

    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.shmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] shmem_init failed")

    print(f"rank[{rank}]: register shmem handler ret={ret}")


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
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    enable_diagnose = args.enable_diagnose
    num_servers = num_ranks // num_local_ranks
    expert_token_nums_type = int(os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", 1))

    """
    # Base number of tokens to be assigned
    base_num_tokens = args.num_tokens

    # Set the fluctuation range (for example, ±10%)
    fluctuation_percentage = 0.1  # 10% fluctuation
    min_fluctuation = 2  # Minimum absolute fluctuation for num_tokens < 10

    # Dynamically calculate num_tokens for each rank
    if base_num_tokens < 10:
        # For small num_tokens, use a combination of proportional and absolute fluctuation
        fluctuation = random.randint(-min_fluctuation, min_fluctuation)
        num_tokens = base_num_tokens + fluctuation
    else:
        # For larger num_tokens, use proportional fluctuation
        fluctuation = random.uniform(1 - fluctuation_percentage, 1 + fluctuation_percentage)
        num_tokens = int(base_num_tokens * fluctuation)

    # Ensure num_tokens is at least 1
    num_tokens = max(num_tokens, 1)
    """

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, num_experts={num_experts}, "
            f"num_ranks={num_ranks}, active_ranks={args.active_ranks}",
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
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
        # topk_idx = torch.zeros((num_tokens, num_topk), dtype=torch.int64, device='npu')
        # for t in range(num_tokens):
        #     start = (t * num_topk) % num_experts
        #     for k in range(num_topk):
        #         topk_idx[t, k] = (start + k) % num_experts

    rank_idx = topk_idx // experts_per_rank
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="npu")
    token_idx_in_rank = torch.full(
        (num_ranks, num_tokens), -1, dtype=torch.long, device="npu"
    )
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(
            count, dtype=torch.long, device="npu"
        )
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = (token_idx_in_rank >= 0).to(torch.int)
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

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
    try:
        assert torch.allclose(
            ref_num_tokens_per_rank, num_tokens_per_rank
        ), f"Assertion num_tokens_per_rank failed on rank {rank}: Expected {num_tokens_per_rank}, Actual {ref_num_tokens_per_rank}"
        assert torch.allclose(
            ref_num_tokens_per_expert, num_tokens_per_expert
        ), f"Assertion num_tokens_per_expert failed on rank {rank}: Expected {num_tokens_per_expert}, Actual {ref_num_tokens_per_expert}"
        assert torch.allclose(
            ref_is_token_in_rank, is_token_in_rank
        ), f"Assertion is_token_in_rank failed on rank {rank}: Expected {is_token_in_rank}, Actual {ref_is_token_in_rank}"
    except AssertionError as e:
        print(e)
        raise
    print(f"{rank=}, dispatch_layout passed", flush=True)

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
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    def get_num_tokens_per_expert_list(rank: int):
        local_expert_token = gbl_num_tokens_per_expert.view(num_ranks, -1)[rank]
        if expert_token_nums_type == 0:
            local_expert_token_list = local_expert_token.cumsum(
                dim=0
            ).tolist()  # 计算前缀和并转为 list
        else:
            local_expert_token_list = local_expert_token.tolist()
        return local_expert_token_list

    def per_token_cast_back_shmem(x_fp8: torch.Tensor, x_scales: torch.Tensor):
        if x_fp8.numel() == 0:
            return x_fp8.to(torch.bfloat16)
        if x_scales.dtype == torch.int:
            x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
            x_scales = x_scales.view(dtype=torch.float)
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
        x_scales = x_scales.view(x_fp8.size(0), -1, 1)

        recv_x_bf16 = (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)

        device = torch.device("npu:{}".format(local_rank))
        shmem_recv_x = symm_mem.empty(
            recv_x_bf16.shape, dtype=torch.bfloat16, device=device
        )
        return shmem_recv_x

    def test_correctness():
        for current_x in filter(lambda elem: elem is not None, (x, x_pure_rand)):
            if local_rank == 0:
                print(
                    f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, with top-k {num_topk} ...',
                    flush=True,
                )
            # Test dispatch
            dispatch_args = {
                "x": current_x,
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
            recv_x = (
                per_token_cast_back_shmem(*recv_x)
                if isinstance(recv_x, tuple)
                else recv_x
            )

            # Checks notify output
            local_expert_token_list = get_num_tokens_per_expert_list(rank)
            assert local_expert_token_list == recv_num_tokens_per_expert_list

            all_recv_count = handle[8]

            total_recv_tokens = sum(local_expert_token_list)
            print(
                f"{rank=}, {total_recv_tokens=} {recv_x.shape}, dispatch passed",
                flush=True,
            )

            # Test combine
            combine_args = {
                "x": recv_x,
                "handle": handle,
                "config": config,
                "async_finish": False,
                "topk_weights": handle[7],
            }
            combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
            check_x = combined_x.float()
            ref_x = x_pure_rand if current_x is x_pure_rand else x

            assert (
                calc_diff(
                    check_x,
                    ref_x
                    * handle[7].masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
                )
                < 5e-5
            )

            del recv_x  # release symmetric tensor

        if local_rank == 0:
            print(" passed", flush=True)

    def test_tuning():
        # Tune dispatch performance
        fp8_factor = (1 + 4 / 128) / 2
        config = deep_ep.Config(24, 8, buffer_size)

        # test layout
        def test_layout_func():
            (
                tmp_num_tokens_per_rank,
                _,
                tmp_num_tokens_per_expert,
                tmp_is_token_in_rank,
                _,
            ) = buffer.get_dispatch_layout(topk_idx, num_experts)

            del tmp_num_tokens_per_expert  # release symmetric tensor

        avg_t, min_t, max_t = bench(partial(test_layout_func))
        print(f"[layout] Kernel performance: {avg_t * 1000:.3f} ms", flush=True)
        print("", flush=True)

        # test dispatch+combine
        current_x = x

        def test_func():
            tune_dispatch_args = {
                "x": current_x,
                "config": config,
                "num_tokens_per_rank": ref_num_tokens_per_rank,
                "is_token_in_rank": ref_is_token_in_rank,
                "num_tokens_per_expert": ref_num_tokens_per_expert,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights,
            }
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_num_tokens_per_expert_list,
                handle,
                event,
            ) = buffer.dispatch(**tune_dispatch_args)

            tune_combine_args = {
                "x": recv_x,
                "handle": handle,
                "config": config,
                "async_finish": False,
                "topk_weights": handle[7],
            }
            combined_x, combined_topk_weights, event = buffer.combine(
                **tune_combine_args
            )

            del recv_x  # release symmetric tensor

        local_expert_token_list = get_num_tokens_per_expert_list(rank)
        real_recv_tokens = sum(local_expert_token_list)
        dispatch_bf16_recv_bytes = real_recv_tokens * hidden * 2
        combine_bf16_send_bytes = dispatch_bf16_recv_bytes

        notify_t, dispatch_t, combine_t = bench_kineto(
            partial(test_func),
            kernel_names=(
                "ShmemNotifyDispatch",
                "ShmemMoeDispatchNormal",
                "ShmemMoeCombineNormal",
            ),
        )
        if local_rank == 0:
            print(
                f"[tuning] Notify avg_t: {notify_t * 1e6:.2f} us | Dispatch {dispatch_bf16_recv_bytes / 1e9 / dispatch_t:.2f} GB/s (HCCS), avg_t: {dispatch_t * 1e6:.2f} us | "
                f"Combine {combine_bf16_send_bytes / 1e9 / combine_t:.2f} GB/s (HCCS), avg_t: {combine_t * 1e6:.2f} us",
                flush=True,
            )
            print("", flush=True)

    test_correctness()
    test_tuning()


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # register_shmem()

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)
    torch.manual_seed(rank)

    test_main(args, num_local_ranks, local_rank, num_ranks, rank, buffer, group)
    if local_rank == 0:
        print("", flush=True)

    # _ = ash.shmem_finialize()
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
        "--num-tokens", type=int, default=1024, help="Number of tokens (default: 4096)"
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
        "--enable-diagnose",
        action="store_true",
        help="Whether to enable diagnose for testing",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
