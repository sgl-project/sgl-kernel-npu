"""Four-rank synthetic regression for sparse final expert IDs; no model weights.

Run with four explicitly selected, idle A3 devices and the installed deep_ep.
This is a correctness test, not a performance benchmark.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def sparse_topk(torch, rank, case, tokens=2048):
    counts126, singleton127 = {
        "last_expert_one": ([4, 4, 1, 6], True),
        "last_expert_empty": ([2, 5, 5, 3], False),
    }[case]
    generator = torch.Generator(device="cpu").manual_seed(812 + rank)
    ids = (
        torch.rand(
            (tokens, 126), generator=generator, device="cpu", dtype=torch.float32
        )
        .topk(8, dim=1)
        .indices
    )
    ids[: counts126[rank], 0] = 126
    if singleton127 and rank == 2:
        ids[counts126[rank], 0] = 127
    return ids


def worker(local_rank, args):
    import torch
    import torch.distributed as dist
    import torch_npu
    from deep_ep import Buffer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_dispatch_ffn_combine import init_base_weights, init_fused2_weights_int8
    from utils import init_dist

    torch.set_num_threads(1)
    torch.manual_seed(917 + local_rank)
    rank, world, group = init_dist(local_rank, 4)
    torch_npu.npu.config.allow_internal_format = True
    hidden, width, experts, topk, tokens = 2048, 1536, 128, 8, 2048
    weights = init_fused2_weights_int8(*init_base_weights(32, hidden, width))
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="npu") * 0.1
    gates = torch.full((tokens, topk), 1 / topk, dtype=torch.float32, device="npu")
    capacity = tokens * topk * world
    buffer = Buffer(
        group,
        num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(
            capacity, hidden, world, experts
        ),
        low_latency_mode=True,
        num_qps_per_rank=32,
    )
    report = {
        "scope": "synthetic sparse-tail correctness; no latency claims",
        "cases": {},
    }
    for case in ("last_expert_one", "last_expert_empty"):
        ids = sparse_topk(torch, rank, case, tokens)
        expected = torch.bincount(ids.flatten(), minlength=experts).to("npu")
        dist.all_reduce(expected, group=group)
        expected = expected[rank * 32 : (rank + 1) * 32]
        ids = ids.to(device="npu", dtype=torch.int32)
        previous = None
        bad_counts = 0
        changes = 0
        max_change = 0.0
        for _ in range(args.repeats):
            output, counts = buffer.fused_deep_moe(
                x, ids, gates, *weights, capacity, experts, 1, 2
            )
            mismatch = int(not torch.equal(counts, expected.to(counts.dtype)))
            delta = (
                (output.float() - previous.float()).abs().max().item()
                if previous is not None
                else 0.0
            )
            metrics = torch.tensor([mismatch, delta], dtype=torch.float32, device="npu")
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group)
            bad_counts += int(metrics[0].item() != 0)
            changes += int(metrics[1].item() != 0)
            max_change = max(max_change, metrics[1].item())
            previous = output.clone()
        report["cases"][case] = {
            "calls": args.repeats,
            "bad_count_calls": bad_counts,
            "changed_transitions": changes,
            "max_change": max_change,
        }
    if rank == 0:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    dist.destroy_process_group()
    assert all(
        not r["bad_count_calls"] and not r["changed_transitions"]
        for r in report["cases"].values()
    ), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if (args.output is not None and args.output.exists()) or args.repeats < 2:
        parser.error("need a new output path and at least two calls")
    selected = [
        int(x) for x in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if x
    ]
    if len(selected) != 4:
        parser.error("select four idle physical devices explicitly")
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(args,), nprocs=4, join=True)


if __name__ == "__main__":
    main()
