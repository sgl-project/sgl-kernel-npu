import argparse
import time

# noinspection PyUnresolvedReferences
import deep_ep
import torch
import torch.distributed as dist
from utils import bench, calc_diff, init_dist, inplace_unique, per_token_cast_back


# noinspection PyShadowingNames
def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}",
            flush=True,
        )

    # Random data
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    rank_idx = topk_idx // (num_experts // num_ranks)
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
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)
    try:
        try:
            return_values = buffer.get_dispatch_layout(topk_idx, num_experts)
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
    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

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

    for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x)):
        if local_rank == 0:
            print(
                f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, with top-k ...',
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
            < 5e-6
        )

        # For later tuning
        dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
        combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

        if local_rank == 0:
            print(" passed", flush=True)
    if local_rank == 0:
        print("", flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x,)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0,):
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)
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
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                    f"{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                    flush=True,
                )
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
                flush=True,
            )
            print("", flush=True)

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1]], dtype=torch.int32, device="npu"
            )
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results)
                for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(
                all_best_fp8_results_list, best_dispatch_results, group=group
            )
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(
        best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size
    )

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0,):
        if nvl_chunk_size > 0:
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_combine_config(num_ranks)
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
                f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                f"{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                flush=True,
            )
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(
            f"[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    print(f"[Rank {rank} | Local rank {local_rank}] Initializing buffer...", flush=True)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )
    print(f"[Rank {rank}] Buffer created OK.", flush=True)
    torch.manual_seed(rank)

    for i in (24,):
        test_main(args, i, local_rank, num_ranks, rank, buffer, group)
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
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(
        test_loop, args=(num_processes, args), nprocs=num_processes
    )
