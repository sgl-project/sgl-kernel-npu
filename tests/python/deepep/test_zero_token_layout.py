import argparse
import os

import deep_ep
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


NUM_EXPERTS = 256
NUM_TOPK = 8


def run_rank(local_rank: int, world_size: int) -> None:
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=local_rank,
        world_size=world_size,
    )
    group = dist.new_group(list(range(world_size)))

    try:
        buffer = deep_ep.Buffer(
            group,
            int(2e9),
            0,
            low_latency_mode=False,
            num_qps_per_rank=1,
        )

        # Match SGLang's DP-attention idle path: only one DP rank owns a token,
        # while every idle rank still participates in DeepEP layout generation.
        num_tokens = 1 if local_rank == 0 else 0
        if num_tokens:
            topk_idx = torch.arange(
                NUM_TOPK, dtype=torch.int64, device="npu"
            ).reshape(1, NUM_TOPK)
        else:
            topk_idx = torch.empty(
                (0, NUM_TOPK), dtype=torch.int64, device="npu"
            )

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, NUM_EXPERTS)

        assert num_tokens_per_rdma_rank is None
        assert tuple(num_tokens_per_rank.shape) == (world_size,)
        assert tuple(is_token_in_rank.shape) == (num_tokens, world_size)

        expected_routes = NUM_TOPK if num_tokens else 0
        assert int(num_tokens_per_expert.sum().item()) == expected_routes
        if num_tokens == 0:
            assert int(num_tokens_per_rank.sum().item()) == 0
            assert is_token_in_rank.numel() == 0

        print(
            f"rank={local_rank} num_tokens={num_tokens} layout passed",
            flush=True,
        )
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test zero-token DeepEP layouts")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="Number of local ranks to spawn (default: 16)",
    )
    args = parser.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29567")
    os.environ.setdefault("DEEP_USE_MODE", "default")
    torch.multiprocessing.spawn(
        run_rank,
        args=(args.num_processes,),
        nprocs=args.num_processes,
        join=True,
    )
