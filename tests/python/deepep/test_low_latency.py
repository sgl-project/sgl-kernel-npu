import argparse
import os
import random
import time
from functools import partial

import torch
import torch.distributed as dist
import torch_npu
from deep_ep import Buffer
from utils import (
    bench,
    bench_kineto,
    calc_diff,
    calculate_avg_stats,
    hash_tensor,
    init_dist,
    per_token_cast_back,
)


def test(
    aligned_num_tokens: int,  # 对齐后的最大token数
    actual_num_tokens: int,  # 当前rank的实际token数，有效token数
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    drop_percent: float,
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

        if drop_percent > 0:
            enable_neg_one = int(os.getenv("MOE_ENABLE_TOPK_NEG_ONE", 0))
            if enable_neg_one == 0:
                print(
                    "[ERROR] The kernel can't support drop_percent larger than 0 when MOE_ENABLE_TOPK_NEG_ONE was"
                    "unset or 0. Please set to 1 and try again"
                )
                assert enable_neg_one == 1
            drop_mask = (
                torch.rand(
                    (actual_num_tokens, num_topk), dtype=torch.float32, device="npu"
                )
                < drop_percent
            )
            topk_idx[:actual_num_tokens] = topk_idx[:actual_num_tokens].masked_fill(
                drop_mask, -1
            )
    topk_weights = torch.zeros(
        (aligned_num_tokens, num_topk), dtype=torch.float32, device="npu"
    )
    if actual_num_tokens > 0:
        topk_weights[:actual_num_tokens] = torch.randn(
            (actual_num_tokens, num_topk), dtype=torch.float32, device="npu"
        ).abs()

    # Check dispatch correctness
    do_check = True
    return_recv_hook = False
    hash_value, num_times = 0, 0

    cumulative_local_expert_recv_stats = torch.zeros(
        (num_local_experts,), dtype=torch.int, device="npu"
    )
    for dispatch_use_fp8 in (True, False):
        packed_recv_x, packed_recv_count, handle, event, hook = (
            buffer.low_latency_dispatch(
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
        )
        simulated_gemm_x = (
            per_token_cast_back(*packed_recv_x) if dispatch_use_fp8 else packed_recv_x
        )

        all_topk_idx = torch.empty(
            (num_ranks, aligned_num_tokens, num_topk),
            dtype=topk_idx.dtype,
            device="npu",
        )
        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

        local_actual_tokens = torch.tensor(
            [actual_num_tokens], dtype=torch.int32, device="npu"
        )
        all_actual_tokens = torch.empty(num_ranks, dtype=torch.int32, device="npu")
        dist.all_gather_into_tensor(all_actual_tokens, local_actual_tokens, group=group)

        for i in range(num_local_experts if do_check else 0):
            expert_id = rank * num_local_experts + i
            temp = aligned_num_tokens / num_local_experts
            recv_count = packed_recv_count[i]
            recv_x = (
                per_token_cast_back(
                    packed_recv_x[0][int(i * temp) : int((i + 1) * temp)],
                    packed_recv_x[1][int(i * temp) : int((i + 1) * temp)],
                )
                if dispatch_use_fp8
                else packed_recv_x[int(i * temp) : int((i + 1) * temp)]
            )
            if i == 0:
                recv_layout_range = handle[1][(i + 1) * num_ranks - 1]
            else:
                recv_layout_range = (
                    handle[1][(i + 1) * num_ranks - 1] - handle[1][i * num_ranks - 1]
                )

            # Check expert indices
            int_mask = (2**32) - 1
            num_valid_tokens = recv_count.item()

            expected_valid_tokens = 0
            for r in range(num_ranks):
                # 获取rank r的实际token数
                r_actual_tokens = all_actual_tokens[r].item()
                # 获取rank r发送给这个专家的token数
                r_topk_idx = all_topk_idx[r, :r_actual_tokens]
                expected_valid_tokens += (r_topk_idx == expert_id).sum().item()

            assert (
                num_valid_tokens == (recv_layout_range & int_mask).item()
            ), f"{num_valid_tokens} != {recv_layout_range & int_mask}.item()"
            assert (
                num_valid_tokens == expected_valid_tokens
            ), f"{num_valid_tokens} != {expected_valid_tokens}"

            if num_valid_tokens == 0:
                continue
            # Check received data
            recv_x = recv_x[:num_valid_tokens]
            recv_x_amin = recv_x[:, :-128].amin(dim=-1)
            assert torch.allclose(
                recv_x_amin, recv_x[:, :-128].amax(dim=-1), equal_nan=True
            )
            if dispatch_use_fp8:
                hash_value ^= hash_tensor(
                    packed_recv_x[0][int(i * temp) : int(i * temp + num_valid_tokens)]
                )
                hash_value ^= hash_tensor(
                    packed_recv_x[1][int(i * temp) : int(i * temp + num_valid_tokens)]
                )
            else:
                hash_value ^= hash_tensor(
                    packed_recv_x[int(i * temp) : int(i * temp + num_valid_tokens)]
                )

        # Check combine correctness
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            packed_recv_count,
        ) = handle

        out = torch.empty(
            (aligned_num_tokens, hidden), dtype=torch.bfloat16, device="npu"
        )
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

        if do_check:
            if actual_num_tokens > 0:
                # 只考虑有效token
                expected_x = torch.zeros(
                    (aligned_num_tokens, hidden), dtype=torch.bfloat16, device="npu"
                )
                expected_x[:actual_num_tokens] = torch.ones(
                    (actual_num_tokens, hidden), dtype=torch.bfloat16, device="npu"
                ) * (rank - rank_offset)
                expected_x[:actual_num_tokens, -128:] = (
                    torch.arange(actual_num_tokens, device="npu")
                    .to(torch.bfloat16)
                    .view(-1, 1)
                )

                diff = calc_diff(
                    expected_x[:actual_num_tokens]
                    * topk_weights[:actual_num_tokens]
                    .masked_fill(topk_idx[:actual_num_tokens] == -1, 0)
                    .sum(dim=1)
                    .view(-1, 1),
                    combined_x[:actual_num_tokens],
                )
                assert torch.isnan(combined_x[:actual_num_tokens]).sum().item() == 0
                if dispatch_use_fp8:
                    assert diff < 1e-4, f"Error: {diff=}"
                else:
                    assert diff < 1e-5, f"Error: {diff=}"
            hash_value ^= hash_tensor(
                combined_x[:actual_num_tokens]
                if actual_num_tokens > 0
                else torch.tensor([], device="npu")
            )

            print(f"rank {rank} PASSED (actual tokens: {actual_num_tokens})")

    # noinspection PyShadowingNames
    def test_func(zero_copy: bool, return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x,
            topk_idx,
            aligned_num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            use_fp8=dispatch_use_fp8,
            async_finish=False,
            return_recv_hook=return_recv_hook,
        )
        combined_x, event, hook = buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            zero_copy=zero_copy,
            return_recv_hook=return_recv_hook,
        )

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden // 128 * 4 + 16), hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(actual_num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(
        partial(test_func, zero_copy=False, return_recv_hook=False)
    )
    print(
        f"[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, "
        f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us, "
        f"actual_tokens={actual_num_tokens}",
        flush=True,
    )

    # Separate profiling
    for return_recv_hook in (False,):
        enable_neg_one = int(os.getenv("MOE_ENABLE_TOPK_NEG_ONE", 0))
        dist.barrier()
        dispatch_t, combine_t = bench_kineto(
            partial(test_func, zero_copy=False, return_recv_hook=return_recv_hook),
            kernel_names=("MoeDistributeDispatchV2", "MoeDistributeCombineV2"),
            barrier_comm_profiling=True,
            suppress_kineto_output=True,
            num_kernels_per_period=2 if return_recv_hook else 1,
            trace_path=None,
        )
        if not return_recv_hook:
            print(
                f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | "
                f"Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us",
                flush=True,
            )
            calculate_avg_stats(
                dispatch_t=dispatch_t,
                num_dispatch_comm_bytes=num_dispatch_comm_bytes,
                combine_t=combine_t,
                num_combine_comm_bytes=num_combine_comm_bytes,
                rank=rank,
                num_ranks=num_ranks,
                root_rank=0,
            )

        else:
            print(
                f"[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | "
                f"Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us",
                flush=True,
            )

    return hash_value


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
    base_num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    use_experts = num_experts if shared_expert_rank_num == 0 else (num_experts - 1)
    use_ranks = num_ranks - shared_expert_rank_num
    drop_percent = args.drop_percent

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

    raw_num_tokens = max(num_tokens, 1)

    local_tokens_tensor = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
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
        num_qps_per_rank=use_experts // use_ranks if use_ranks > 0 else 1,
    )

    test(
        aligned_num_tokens,
        raw_num_tokens,
        hidden,
        use_experts,
        num_topk,
        rank,
        use_ranks,
        group,
        buffer,
        drop_percent,
        seed=1,
    )

    do_pressure_test = args.pressure_test
    for seed in range(int(1e9) if do_pressure_test else 0):
        if rank == 0:
            print(f"Testing with seed {seed} ...", flush=True)
        ref_hash = test(
            aligned_num_tokens,
            raw_num_tokens,
            hidden,
            use_experts,
            num_topk,
            rank,
            use_ranks,
            group,
            buffer,
            drop_percent,
            seed=seed,
        )
        for i in range(20):
            assert (
                test(
                    aligned_num_tokens,
                    raw_num_tokens,
                    hidden,
                    use_experts,
                    num_topk,
                    rank,
                    use_ranks,
                    group,
                    buffer,
                    drop_percent,
                    seed=seed,
                )
                == ref_hash
            ), f"Error: seed={seed}"
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
        "--pressure-test", action="store_true", help="Whether to do pressure test"
    )
    parser.add_argument(
        "--drop-percent",
        type=float,
        default=0.0,
        help="Percentage of dropping an individual top-k index (set to -1). ",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
