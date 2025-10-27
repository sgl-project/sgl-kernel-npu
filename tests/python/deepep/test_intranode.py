import argparse
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

MAX_BATCH_SIZE = 4096
enable_a2_test = False

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
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    rank_idx = topk_idx // experts_per_rank
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Server meta
    if enable_a2_test:
        count_num_expert = [0] * num_experts
        num_tokens_per_server_uniq = torch.zeros((num_servers,), dtype=torch.int, device="npu")
        num_each_token_to_server = torch.zeros((num_tokens * num_servers,), dtype=torch.int, device="npu")
        each_token_to_num_server = torch.zeros((num_tokens,), dtype=torch.int, device="npu")
        each_token_offset_to_server = torch.full((num_tokens * num_servers,), -1, dtype=torch.int, device="npu")
        send_token_idx = torch.full((num_tokens * num_experts,), -1, dtype=torch.int, device="npu")
        expert_rank_token_idx = torch.zeros((num_experts * MAX_BATCH_SIZE,), dtype=torch.int, device="npu")

        for i in range(num_tokens):
            seen_server = [0] * num_servers
            for j in range(num_topk):
                expert_id = topk_idx[i][j]
                rank_id = expert_id // experts_per_rank
                server_id = rank_id // num_local_ranks
                if seen_server[server_id] == 0:
                    each_token_offset_to_server[i * num_servers + server_id] = num_tokens_per_server_uniq[server_id]
                    num_tokens_per_server_uniq[server_id] += 1
                    each_token_to_num_server[i] += 1
                    seen_server[server_id] += 1
                num_each_token_to_server[i * num_servers + server_id] += 1
                send_token_idx[i * num_experts + expert_id] = count_num_expert[expert_id]
                count_num_expert[expert_id] += 1

        count_num_expert = [0] * num_experts
        for i in range(num_tokens):
            for j in range(num_topk):
                expert_id = topk_idx[i][j]
                rank_id = expert_id // experts_per_rank
                server_id = rank_id // num_local_ranks
                expert_rank_token_idx[expert_id * MAX_BATCH_SIZE + count_num_expert[expert_id]] = \
                each_token_offset_to_server[i * num_servers + server_id]
                count_num_expert[expert_id] += 1

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
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)
    try:
        try:
            return_values = buffer.get_dispatch_layout(topk_idx, num_experts)
            if enable_a2_test:
                notify_send_data = buffer.get_notify_send_data()
                ref_num_tokens_per_server_uniq = notify_send_data[num_experts:num_experts + num_servers]
                ref_num_each_token_to_server = \
                    notify_send_data[num_experts + num_servers:num_experts + num_servers * (1 + num_tokens)]
                ref_each_token_to_num_server = \
                    notify_send_data[num_experts + num_servers * (1 + MAX_BATCH_SIZE):\
                                     num_experts + num_servers + MAX_BATCH_SIZE * num_servers + num_tokens]
                ref_each_token_offset_to_server = \
                    notify_send_data[num_experts + num_servers + MAX_BATCH_SIZE * (num_servers + 1):\
                                     num_experts + num_servers + MAX_BATCH_SIZE * (num_servers + 1) +\
                                     num_servers * num_tokens]
                ref_send_token_idx = \
                    notify_send_data[num_experts + num_servers + MAX_BATCH_SIZE * (num_servers * 2 + 1):\
                                     num_experts + num_servers + MAX_BATCH_SIZE * (num_servers * 2 + 1) +\
                                     num_tokens * num_experts]
                ref_expert_rank_token_idx = \
                    notify_send_data[num_experts + num_servers + MAX_BATCH_SIZE * (num_servers * 2 + num_experts + 1):\
                                     num_experts + num_servers + MAX_BATCH_SIZE *\
                                     (num_servers * 2 + num_experts + num_experts + 1)]
        except Exception as e:
            print(f"Error occurred while calling get_dispatch_layout: {e}")
            raise

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
            if enable_a2_test:
                assert torch.allclose(
                    num_tokens_per_expert, notify_send_data[:num_experts]
                ), f"Assertion num_tokens_per_rank failed on rank {rank}: Expected {ref_num_tokens_per_expert}, Actual {notify_send_data[:num_experts]}"
                assert torch.allclose(
                    num_tokens_per_server_uniq, ref_num_tokens_per_server_uniq
                ), f"Assertion num_tokens_per_server_uniq failed on rank {rank}: Expected {num_tokens_per_server_uniq}, Actual {ref_num_tokens_per_server_uniq}"
                assert torch.allclose(
                    num_each_token_to_server, ref_num_each_token_to_server
                ), f"Assertion num_each_token_to_server failed on rank {rank}: Expected {num_each_token_to_server}, Actual {ref_num_each_token_to_server}"
                assert torch.allclose(
                    each_token_to_num_server, ref_each_token_to_num_server
                ), f"Assertion each_token_to_num_server failed on rank {rank}: Expected {each_token_to_num_server}, Actual {ref_each_token_to_num_server}"
                assert torch.allclose(
                    each_token_offset_to_server, ref_each_token_offset_to_server
                ), f"Assertion each_token_offset_to_server failed on rank {rank}: Expected {each_token_offset_to_server}, Actual {ref_each_token_offset_to_server}"
                assert torch.allclose(
                    send_token_idx, ref_send_token_idx
                ), f"Assertion send_token_idx failed on rank {rank}: Expected {send_token_idx}, Actual {ref_send_token_idx}"
                assert torch.allclose(
                    expert_rank_token_idx, ref_expert_rank_token_idx
                ), f"Assertion expert_rank_token_idx failed on rank {rank}: Expected {expert_rank_token_idx}, Actual {ref_expert_rank_token_idx}"
        except AssertionError as e:
            print(e)
            raise
    except Exception as e:
        print(f"An error occurred: {e}")

    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
    print("", flush=True)
    dist.barrier()
    time.sleep(1)

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

    # Test diagnose function
    # noinspection PyShadowingNames
    def test_diagnose(
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        combine_send_cost_stats: Optional[torch.Tensor] = None,
    ):
        for current_x in filter(lambda elem: elem is not None, (x_pure_rand,)):
            dispatch_args = {
                "x": current_x,
                "num_tokens_per_rank": num_tokens_per_rank,
                "is_token_in_rank": is_token_in_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "config": config,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights_pure_rand,
                "dispatch_wait_recv_cost_stats": dispatch_wait_recv_cost_stats,
            }
            if dispatch_wait_recv_cost_stats is not None:
                bench(lambda: buffer.dispatch(**dispatch_args), num_warmups=0)
            if combine_send_cost_stats is not None:
                (
                    recv_x,
                    recv_topk_idx,
                    recv_topk_weights,
                    recv_num_tokens_per_expert_list,
                    handle,
                    event,
                ) = buffer.dispatch(**dispatch_args)
                recv_x = (
                    per_token_cast_back(*recv_x)
                    if isinstance(recv_x, tuple)
                    else recv_x
                )
                combine_args = {
                    "x": recv_x,
                    "handle": handle,
                    "topk_weights": handle[7],
                    "config": config,
                    "async_finish": False,
                    "combine_send_cost_stats": combine_send_cost_stats,
                }
                bench(lambda: buffer.combine(**combine_args), num_warmups=0)
        for stats, title in (
            (dispatch_wait_recv_cost_stats, "Dispatch wait recv cost"),
            (combine_send_cost_stats, "Combine send cost"),
        ):
            if stats is None:
                continue
            gather_list = (
                [torch.zeros_like(stats) for _ in range(group.size())]
                if rank == 0
                else None
            )
            dist.gather(stats, gather_list=gather_list, group=group, dst=0)
            if rank == 0:
                stats_mat = torch.stack(gather_list, dim=0)
                print(f"{title} stats:")
                print(stats_mat)
                res = diagnose_matrix(
                    stats_mat, thres_col=1.0, thres_row=2.0, thres_point=5.0
                )
                print(
                    f"[Diagnose {title}] abnormal_rows {res['abnormal_rows']}, "
                    f"abnormal_cols {res['abnormal_cols']}, abnormal_points {res['abnormal_points']}"
                )

    for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x)):
        if local_rank == 0:
            print(
                f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, with top-k {num_topk} ...',
                flush=True,
            )
        dispatch_args = {
            "x": current_x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
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
        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

        # Checks
        rank_prefix_matrix = handle[0]
        # todo 1. Duplicate tansmission to experts of the same rank.
        # assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
        # todo 2. recv_num_tokens_per_expert_list is the prefix sum of the actual data.
        # assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
        # todo 3. empty return value rank_prefix_matrix
        # if current_x is not x_pure_rand:
        #     check_data(recv_x, rank_prefix_matrix)

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
                ref_x * handle[7].masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1),
            )
            < 5e-5
        )

        # For later tuning
        dispatch_bf16_recv_bytes = recv_x.numel() * 2
        combine_bf16_send_bytes = dispatch_bf16_recv_bytes

        if local_rank == 0:
            print(" passed", flush=True)
    if local_rank == 0:
        print("", flush=True)

    # Tune dispatch performance
    fp8_factor = (1 + 4 / 128) / 2
    config = deep_ep.Config(24, 8, buffer_size)
    for current_x in filter(lambda elem: elem is not None, (x,)):
        recv_bytes = (
            (dispatch_bf16_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_recv_bytes
        )

        tune_args = {
            "x": current_x,
            "config": config,
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
        }

        t = bench(lambda: buffer.dispatch(**tune_args))[0]
        if local_rank == 0:
            print(
                f'[tuning] Dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}) {recv_bytes / 1e9 / t:.2f} GB/s (HCCS), avg_t: {t * 1e6:.2f} us',
                flush=True,
            )
            print("", flush=True)

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
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
        "topk_weights": handle[7],
    }
    t = bench(lambda: buffer.combine(**tune_args))[0]
    if local_rank == 0:
        print(
            f"[tuning] Combine {combine_bf16_send_bytes / 1e9 / t:.2f} GB/s (HCCS), avg_t: {t * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)

    # Diagnose test
    if enable_diagnose:
        dispatch_wait_recv_cost_stats = torch.zeros(
            (num_ranks,), dtype=torch.int32, device="npu"
        )
        combine_send_cost_stats = torch.zeros(
            (num_ranks,), dtype=torch.int32, device="npu"
        )
        test_diagnose(
            dispatch_wait_recv_cost_stats=dispatch_wait_recv_cost_stats,
            combine_send_cost_stats=combine_send_cost_stats,
        )


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
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
        "--enable-diagnose",
        action="store_true",
        help="Whether to enable diagnose for testing",
    )
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
