import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import deep_ep
from utils import bench, init_dist, per_token_cast_back


def test_loop(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(
        group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1
    )

    num_tokens, hidden, num_experts, num_topk = 128, 7168, 64, 8
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="npu")
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, num_topk), dtype=torch.int64, device="npu"
    )
    topk_weights = torch.rand(num_tokens, num_topk, dtype=torch.float, device="npu")

    layout = buffer.get_dispatch_layout(topk_idx, num_experts)
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = layout

    config = deep_ep.Config(24, 8, 256)

    env_int8 = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") == "1"
    mode_name = "INT8" if env_int8 else "BF16"
    print(
        f"[rank {rank}] DEEP_NORMAL_MODE_USE_INT8_QUANT={'1' if env_int8 else '0'}",
        flush=True,
    )
    print(f"[rank {rank}] calling dispatch WITHOUT quant_mode kwarg ...", flush=True)

    # KEY: quant_mode not passed -> None -> env var fallback branch
    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": config,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
    }

    # Correctness call (quant_mode omitted -> None -> env var fallback)
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    is_tuple = isinstance(recv_x, tuple)
    mode = "INT8 (tuple recv)" if is_tuple else "BF16 (plain tensor recv)"
    print(f"[rank {rank}] dispatch ran as: {mode}", flush=True)
    print(f"[rank {rank}] env var took effect: {is_tuple == env_int8}", flush=True)

    # Compute received bytes for bandwidth (data + scales if quantized)
    if is_tuple:
        dispatch_recv_bytes = (
            recv_x[0].numel() * recv_x[0].element_size()
            + recv_x[1].numel() * recv_x[1].element_size()
        )
    else:
        dispatch_recv_bytes = recv_x.numel() * recv_x.element_size()
    combine_send_bytes = dispatch_recv_bytes

    # Tune dispatch performance
    t = bench(lambda: buffer.dispatch(**dispatch_args))[0]
    if rank == 0:
        print(
            f"[tuning] Dispatch ({mode_name}) {dispatch_recv_bytes / 1e9 / t:.2f} GB/s (HCCS), "
            f"avg_t: {t * 1e6:.2f} us",
            flush=True,
        )

    # Fresh dispatch to get a valid handle for combine
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)
    recv_x = per_token_cast_back(*recv_x) if is_tuple else recv_x

    # Tune combine performance
    combine_args = {
        "x": recv_x,
        "handle": handle,
        "config": config,
        "async_finish": False,
        "topk_weights": handle[7],
    }
    t = bench(lambda: buffer.combine(**combine_args))[0]
    if rank == 0:
        print(
            f"[tuning] Combine ({mode_name}) {combine_send_bytes / 1e9 / t:.2f} GB/s (HCCS), "
            f"avg_t: {t * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=4)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        test_loop, args=(args.num_processes, args), nprocs=args.num_processes
    )
