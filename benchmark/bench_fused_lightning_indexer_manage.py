"""Native LIM launch benchmark; requires only an A2/A3 sgl-kernel-npu wheel."""

import argparse
import json

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import fused_lightning_indexer_manage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cold", action="store_true")
    args = parser.parse_args()
    if (
        args.batch < 1
        or args.length % 128
        or args.length < args.capacity
        or args.iterations < 1
    ):
        parser.error(
            "batch/iterations must be positive; length must be page aligned and >= capacity"
        )
    if not 2048 <= args.capacity <= 32640 or args.capacity % 128:
        parser.error("capacity must be a multiple of 128 in [2048, 32640]")
    torch.npu.set_device(0)
    batch, heads, length = args.batch, 32, args.length
    device = "npu:0"
    q = torch.randn(batch, heads, 128, dtype=torch.bfloat16, device=device)
    key = torch.randn(length // 128, 128, 1, 128, dtype=q.dtype, device=device)
    vector = lambda value: torch.full((batch,), value, dtype=torch.int32, device=device)
    inputs = [
        torch.ones(batch, heads, dtype=q.dtype, device=device),
        torch.ones(batch, heads, device=device),
        q,
        torch.ones(key.shape[:3], device=device),
        key,
        torch.arange(length // 128, dtype=torch.int32, device=device).repeat(batch, 1),
        torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        vector(length),
        vector(length),
        vector(args.capacity),
        vector(-2),
        torch.arange(batch, dtype=torch.int32, device=device),
        torch.full((batch, length), -1, dtype=torch.int32, device=device),
        torch.empty(batch, 1, 2048, dtype=torch.int32, device=device),
        torch.empty(batch, 1, 2048, dtype=torch.int32, device=device),
        vector(0),
        torch.empty(batch, 32768, dtype=torch.int32, device=device),
        torch.empty(batch, 32768, dtype=torch.int32, device=device),
        vector(0),
    ]
    for _ in range(5):
        fused_lightning_indexer_manage(*inputs)
    if not args.cold:
        inputs[10].fill_(-1)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fused_lightning_indexer_manage(*inputs)
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(args.iterations):
        graph.replay()
    end.record()
    end.synchronize()
    print(
        json.dumps(
            {
                "batch": batch,
                "length": length,
                "capacity": args.capacity,
                "cold": args.cold,
                "launch_us": start.elapsed_time(end) * 1000 / args.iterations,
                "miss_counts": inputs[18].cpu().tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
