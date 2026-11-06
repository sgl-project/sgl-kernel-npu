"""A2 dynamic INT8 dispatch regression; run on two nodes with eight NPUs each.

On each node, with NODE_RANK set to 0 or 1 and MASTER_ADDR set to node 0:

    HCCL_BUFFSIZE=3000 HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 \
    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=$NODE_RANK \
        --master-addr=$MASTER_ADDR --master-port=29500 \
        tests/python/deepep/test_internode_a2_int8.py

Use an A2 DeepEP build and leave HCCL_OP_EXPANSION_MODE unset. Clearing both
HCCL_INTRA_* variables exercises the alternative dynamic INT8 tiling key.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def make_inputs(rank, world_size, num_tokens, hidden, iteration):
    """Build reproducible CPU inputs, including local and remote expert routes."""
    generator = torch.Generator().manual_seed(1000 * iteration + rank)
    x = torch.randint(-127, 128, (num_tokens, hidden), generator=generator).float()
    x[:, 0], x[:, 1] = -127, 127
    token = torch.arange(num_tokens)
    # Power-of-two amplitudes keep normal rows away from rounding boundaries.
    x *= (2.0 ** ((token + rank + iteration) % 9 - 4)).unsqueeze(1)
    kind = token % 8
    x[kind == 0] = 0
    x[kind == 1] = x[kind == 1].abs()
    x[kind == 2] = -x[kind == 2].abs()
    x[kind == 4] = 1
    x[kind == 5] = -1
    x[kind == 6] = torch.where(x[kind == 6] >= 0, 1.0e-32, -1.0e-32)
    x[kind == 7] = 2.0**-133  # Smallest positive BF16 subnormal.

    # Two experts per destination, including this node and the next node.
    destinations = (rank + token[:, None] + torch.tensor([0, 1, 8, 9])) % world_size
    topk_idx = (2 * destinations[:, :, None] + torch.arange(2)).reshape(-1, 8)
    weights = (rank + 1) / 32 + token[:, None] / 128 + torch.arange(8)[None, :] / 16
    return x.to(torch.bfloat16), topk_idx, weights


def dispatch_reference(rank, world_size, num_tokens, hidden, iteration):
    """Order rows by local expert, source rank and source token on the CPU."""
    inputs = [
        make_inputs(src, world_size, num_tokens, hidden, iteration)
        for src in range(world_size)
    ]
    rows, weights, counts = [], [], []
    for expert in (2 * rank, 2 * rank + 1):
        count = 0
        for x, topk_idx, topk_weights in inputs:
            tokens, slots = torch.where(topk_idx == expert)
            rows.append(x[tokens])
            weights.append(topk_weights[tokens, slots])
            count += tokens.numel()
        counts.append(count)

    x = torch.cat(rows).float()
    absmax = x.abs().amax(dim=1).clamp_min(1.0e-30)
    scales = absmax / 127
    quantized = (x * (127 / absmax).unsqueeze(1)).round().to(torch.int8)
    return quantized, scales, torch.cat(weights), counts


def check_dispatch(recv_x, recv_weights, counts, expected):
    assert isinstance(recv_x, tuple), "Dispatch must return INT8 tokens and scales"
    quantized, scales = (tensor.cpu() for tensor in recv_x)
    expected_x, expected_scales, expected_weights, expected_counts = expected
    assert torch.isfinite(scales).all() and (scales > 0).all()
    torch.testing.assert_close(quantized, expected_x, rtol=0, atol=0)
    # No absolute tolerance: it would hide missing scales for tiny/zero rows.
    torch.testing.assert_close(scales, expected_scales, rtol=2e-6, atol=0)
    torch.testing.assert_close(recv_weights.cpu(), expected_weights, rtol=0, atol=0)
    if os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", "1") == "0":
        expected_counts = torch.tensor(expected_counts).cumsum(0).tolist()
    assert counts == expected_counts, (counts, expected_counts)


def run_case(buffer, rank, world_size, num_tokens, hidden, iteration):
    x, topk_idx, weights = (
        tensor.npu()
        for tensor in make_inputs(rank, world_size, num_tokens, hidden, iteration)
    )
    per_rank, per_rdma_rank, per_expert, in_rank, _ = buffer.get_dispatch_layout(
        topk_idx, world_size * 2
    )
    recv_x, _, _, counts, handle, _ = buffer.dispatch(
        x,
        num_tokens_per_rank=per_rank,
        num_tokens_per_rdma_rank=per_rdma_rank,
        num_tokens_per_expert=per_expert,
        is_token_in_rank=in_rank,
        topk_idx=topk_idx,
        topk_weights=weights,
        use_fp8=True,  # The public API selects dynamic INT8 on A2.
    )
    torch.npu.synchronize()
    expected = dispatch_reference(rank, world_size, num_tokens, hidden, iteration)
    # handle[8] holds the weights actually unpacked from the transmitted tokens.
    check_dispatch(recv_x, handle[8], counts, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 256, 7168])
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[8, 65])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if any(h <= 0 or h > 7168 or h % 32 for h in args.hidden):
        parser.error("--hidden must be a multiple of 32 in [32, 7168]")
    if any(n < 8 or n > 4096 for n in args.num_tokens):
        parser.error("--num-tokens must be in [8, 4096] to cover all input patterns")

    import deep_ep
    import torch_npu  # noqa: F401

    if int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 8:
        parser.error("Launch with torchrun and eight processes per A2 node")
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 16 or world_size % 8:
        parser.error("At least two complete A2 nodes are required")
    if os.getenv("DEEP_USE_MODE", "default") != "default":
        parser.error("Unset DEEP_USE_MODE so the A2 kernel is exercised")

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        buffer = deep_ep.Buffer(dist.group.WORLD, int(2e9), 0, num_qps_per_rank=1)
        assert buffer.runtime.get_num_rdma_ranks() == world_size // 8, (
            "This regression requires the A2 internode implementation"
        )
        for hidden in args.hidden:
            for num_tokens in args.num_tokens:
                # Repeat with new data to catch stale scratch/metadata reuse.
                for iteration in range(args.repeat):
                    run_case(buffer, rank, world_size, num_tokens, hidden, iteration)
                if rank == 0:
                    print(f"Passed: tokens={num_tokens}, hidden={hidden}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
