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
    num_tokens: int,  # 当前rank的实际token数，有效token数
    hidden: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    buffer: Buffer,
    seed: int = 0,
    quant_type: str = "no",
    local_rank: int = 0,
):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), "Too many ranks (exceeding test precision limit)"

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (
        rank - rank_offset
    )
    x[:, -128:] = torch.arange(num_tokens, device="npu").to(torch.bfloat16).view(-1, 1)
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="npu")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]

    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="npu"
    ).abs()

    # Check dispatch correctness
    do_check = True
    return_recv_hook = False
    hash_value, num_times = 0, 0

    cumulative_local_expert_recv_stats = torch.zeros(
        (num_local_experts,), dtype=torch.int, device="npu"
    )

    if quant_type == "int8":
        quant_configs = [(True, False, False)]
    else:  # no quant
        quant_configs = [(False, False, False)]

    for dispatch_use_fp8, dispatch_use_ue8m0, dispatch_use_mxfp4 in quant_configs:
        for current_x in filter(lambda elem: elem is not None, (x_pure_rand,)):
            if local_rank == 0:
                print(
                    f'[testing] Running with {quant_type=}, data={"rand" if current_x is x_pure_rand else "uniform"} ...',
                    flush=True,
                )

            packed_recv_x, packed_recv_count, handle, event, hook = (
                buffer.low_latency_dispatch(
                    current_x,
                    topk_idx,
                    aligned_num_tokens,
                    num_experts,
                    use_fp8=dispatch_use_fp8,
                    round_scale=False,
                    use_ue8m0=dispatch_use_ue8m0,
                    use_mxfp4=dispatch_use_mxfp4,
                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                    async_finish=not return_recv_hook,
                    return_recv_hook=return_recv_hook,
                    topk_weights=topk_weights,
                )
            )
            simulated_gemm_x = (
                per_token_cast_back(*packed_recv_x)
                if dispatch_use_fp8
                else packed_recv_x
            )

            padding_size = aligned_num_tokens - num_tokens
            if padding_size > 0:
                padding_tensor = torch.full(
                    (padding_size, num_topk),
                    fill_value=-1,
                    dtype=topk_idx.dtype,
                    device="npu",
                )
                topk_idx_padded = torch.cat([topk_idx, padding_tensor], dim=0)
            else:
                topk_idx_padded = topk_idx

            all_topk_idx = torch.empty(
                (num_ranks, aligned_num_tokens, num_topk),
                dtype=topk_idx.dtype,
                device="npu",
            )
            dist.all_gather_into_tensor(all_topk_idx, topk_idx_padded, group=group)

        # Check combine correctness
        src_info = handle[0]
        layout_range = handle[1]
        num_max_dispatch_tokens_per_rank = handle[2]
        hidden = handle[3]
        packed_recv_count = handle[5]
        expand_scales = handle[6]

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
            ref_x = x_pure_rand if current_x is x_pure_rand else x
            diff = calc_diff(
                ref_x
                * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
                combined_x,
            )
            assert torch.isnan(combined_x).sum().item() == 0
            golden = ref_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(
                dim=1
            ).view(-1, 1)
            eps = 1e-8
            golden_nozero = torch.where(golden == 0, eps, golden)
            max_diff = torch.max(torch.abs(combined_x - golden) / golden_nozero).item()
            avg_diff = torch.mean(torch.abs(combined_x - golden) / golden_nozero).item()
            print(
                f"rank {rank} PASSED [{quant_type=}] avg_diff={avg_diff:.5f}, max_diff={max_diff:.5f}, cosine_diff={diff:.5f}"
            )
            if dispatch_use_mxfp4:
                assert diff < 1e-2, f"Error: {diff=}"
            elif dispatch_use_ue8m0:
                assert diff < 1e-3, f"Error: {diff=}"
            elif dispatch_use_fp8:
                assert diff < 1e-4, f"Error: {diff=}"
            else:
                assert diff < 1e-5, f"Error: {diff=}"
            hash_value ^= hash_tensor(combined_x)
            if local_rank == 0:
                print(" passed", flush=True)
    if local_rank == 0:
        print("", flush=True)

    # noinspection PyShadowingNames
    def test_func(zero_copy: bool, return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            current_x,
            topk_idx,
            aligned_num_tokens,
            num_experts,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            use_fp8=dispatch_use_fp8,
            use_ue8m0=dispatch_use_ue8m0,
            use_mxfp4=dispatch_use_mxfp4,
            async_finish=False,
            return_recv_hook=return_recv_hook,
            topk_weights=topk_weights,
        )
        simulated_gemm_x_local = (
            per_token_cast_back(*recv_x) if dispatch_use_fp8 else recv_x
        )
        combined_x, event, hook = buffer.low_latency_combine(
            simulated_gemm_x_local,
            topk_idx,
            topk_weights,
            handle,
            zero_copy=zero_copy,
            return_recv_hook=return_recv_hook,
        )

    # Calculate bandwidth based on quant_type
    def calculate_dispatch_bytes(num_tokens, hidden, quant_type):
        BLOCK_SIZE = 32
        num_values = num_tokens * hidden
        if quant_type == "int8":
            data_bytes = num_values * 1
            scale_bytes = num_tokens * 2
            return data_bytes + scale_bytes
        else:
            return num_values * 2

    num_bf16_bytes = hidden * 2
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += calculate_dispatch_bytes(
            num_selections, hidden, quant_type
        )
        num_combine_comm_bytes += num_bf16_bytes * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(
        partial(test_func, zero_copy=False, return_recv_hook=False)
    )
    print(f"[test] finish.")
    # return

    # tuning dispatch
    dispatch_args = {
        "x": current_x,
        "topk_idx": topk_idx,
        "num_max_dispatch_tokens_per_rank": aligned_num_tokens,
        "num_experts": num_experts,
        "cumulative_local_expert_recv_stats": cumulative_local_expert_recv_stats,
        "use_fp8": dispatch_use_fp8,
        "use_ue8m0": dispatch_use_ue8m0,
        "use_mxfp4": dispatch_use_mxfp4,
        "topk_weights": topk_weights,
    }
    # dispatch_t = bench(lambda: buffer.low_latency_dispatch(**dispatch_args))[0]
    dispatch_alltoall_t = bench_kineto(
        lambda: buffer.low_latency_dispatch(**dispatch_args),
        kernel_names=("MoeInitRoutingV3", "hcom_alltoallv_"),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
        num_kernels_per_period=2 if return_recv_hook else 1,
        trace_path=None,
    )
    dispatch_t = sum(dispatch_alltoall_t)

    # tuning combine
    recv_x, _, handle, _, _ = buffer.low_latency_dispatch(**dispatch_args)
    simulated_gemm_x_local = (
        per_token_cast_back(*recv_x) if dispatch_use_fp8 else recv_x
    )
    combine_args = {
        "x": simulated_gemm_x_local,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "handle": handle,
    }
    # combine_t = bench(lambda: buffer.low_latency_combine(**combine_args))[0]
    combine_alltoall_t = bench_kineto(
        lambda: buffer.low_latency_combine(**combine_args),
        kernel_names=("hcom_alltoallv_", "MoeFinalizeRoutingV2"),
        barrier_comm_profiling=True,
        suppress_kineto_output=True,
        num_kernels_per_period=2 if return_recv_hook else 1,
        trace_path=None,
    )
    combine_t = sum(combine_alltoall_t)

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

    return hash_value


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    shared_expert_rank_num = int(os.getenv("MOE_SHARED_EXPERT_RANK_NUM", 0))
    base_num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    use_experts = num_experts if shared_expert_rank_num == 0 else (num_experts - 1)
    use_ranks = num_ranks - shared_expert_rank_num

    enable_dynamic_tokens = args.enable_dynamic_tokens

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

        raw_num_tokens = max(num_tokens, 1)
    else:
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
        num_qps_per_rank=use_experts // use_ranks if use_ranks > 0 else 1,
        low_latency_strategy="alltoall",
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
        seed=1,
        quant_type=args.quant_type,
        local_rank=local_rank,
    )
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
        "--enable-dynamic-tokens",
        action="store_true",
        help="Enable dynamic and inconsistent num_tokens across different ranks",
    )
    parser.add_argument(
        "--quant-type",
        dest="quant_type",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="Quantization type for dispatch: bf16, int8 (per-token)",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
