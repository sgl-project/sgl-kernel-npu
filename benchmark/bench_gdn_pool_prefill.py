import argparse
import gc
import json
import math
import statistics
import time

import sgl_kernel_npu  # noqa: F401
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[math.ceil(0.95 * len(samples)) - 1],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def legacy_step(data, pool, h):
    out, final = torch.ops.npu.chunk_gated_delta_rule(
        data.q,
        data.k,
        data.v,
        beta=data.beta,
        initial_state=pool[data.dst_slots],
        actual_seq_lengths=data.actual,
        scale=data.scale,
        chunk_state=h,
        g=data.g,
    )
    pool[data.dst_slots] = final
    return out


def pool_step(data, pool, h):
    return torch.ops.npu.chunk_gated_delta_rule_pool(
        data.q,
        data.k,
        data.v,
        pool,
        data.src_slots,
        data.dst_slots,
        beta=data.beta,
        actual_seq_lengths=data.actual,
        scale=data.scale,
        chunk_state=h,
        g=data.g,
        has_initial_state=data.has_initial,
    )


def prepare(data, pool):
    if data.workload == "cached":
        pool[data.dst_slots] = pool[data.src_slots]
    elif data.workload == "chunked":
        pool[data.dst_slots] = data.initial_pool[data.dst_slots]
    else:
        pool[data.dst_slots] = 0


def peak_allocated(data, pool, h, step):
    prepare(data, pool)
    gc.collect()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    step(data, pool, h)
    torch.npu.synchronize()
    return {
        "incremental_peak_bytes": torch.npu.max_memory_allocated() - before,
        "retained_bytes_after_call": torch.npu.memory_allocated() - before,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare gather + legacy GDN op + scatter with indexed pool I/O."
    )
    parser.add_argument(
        "--workload", choices=["cached", "chunked", "fresh"], default="cached"
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--suffix-length", type=int, default=128)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batch, args.suffix_length, args.warmup) < 1 or args.rounds < 2:
        raise ValueError(
            "batch, suffix length and warmup must be positive; rounds >= 2"
        )

    torch.npu.set_device("npu:0")
    torch.manual_seed(args.seed)
    device = torch.device("npu:0")
    batch = args.batch
    total = batch * args.suffix_length
    data = argparse.Namespace(
        workload=args.workload,
        q=F.normalize(
            torch.randn(total, args.key_heads, args.head_dim, device=device), dim=-1
        ).to(torch.bfloat16),
        k=F.normalize(
            torch.randn(total, args.key_heads, args.head_dim, device=device), dim=-1
        ).to(torch.bfloat16),
        v=0.1
        * torch.randn(
            total,
            args.value_heads,
            args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        g=F.logsigmoid(torch.randn(total, args.value_heads, device=device)).float(),
        beta=torch.rand(total, args.value_heads, dtype=torch.bfloat16, device=device),
        actual=torch.full(
            (batch,), args.suffix_length, dtype=torch.int32, device=device
        ),
        scale=args.head_dim**-0.5,
    )
    data.dst_slots = torch.arange(17, 17 + batch, dtype=torch.int32, device=device)
    data.src_slots = (
        torch.full((batch,), 5, dtype=torch.int32, device=device)
        if args.workload == "cached"
        else data.dst_slots
    )
    data.has_initial = torch.full(
        (batch,), args.workload != "fresh", dtype=torch.bool, device=device
    )
    data.initial_pool = 0.05 * torch.randn(
        max(64, 18 + batch),
        args.value_heads,
        args.head_dim,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    legacy_pool = data.initial_pool.clone()
    indexed_pool = data.initial_pool.clone()
    h_shape = (
        batch * ((args.suffix_length + 63) // 64),
        args.value_heads,
        args.head_dim,
        args.head_dim,
    )
    legacy_h = torch.empty(h_shape, dtype=torch.bfloat16, device=device)
    indexed_h = torch.empty_like(legacy_h)

    prepare(data, legacy_pool)
    prepare(data, indexed_pool)
    expected_out = legacy_step(data, legacy_pool, legacy_h)
    actual_out = pool_step(data, indexed_pool, indexed_h)
    torch.npu.synchronize()
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(indexed_h, legacy_h, rtol=0, atol=0)
    torch.testing.assert_close(indexed_pool, legacy_pool, rtol=0, atol=0)
    if args.workload == "cached":
        torch.testing.assert_close(
            indexed_pool[data.src_slots],
            data.initial_pool[data.src_slots],
            rtol=0,
            atol=0,
        )

    paths = [
        ("legacy", legacy_pool, legacy_h, legacy_step),
        ("pool", indexed_pool, indexed_h, pool_step),
    ]
    for _ in range(args.warmup):
        for _, pool, h, step in paths:
            prepare(data, pool)
            step(data, pool, h)
    torch.npu.synchronize()
    samples = [[], []]
    for round_index in range(args.rounds):
        order = (0, 1) if round_index % 2 == 0 else (1, 0)
        for index in order:
            _, pool, h, step = paths[index]
            prepare(data, pool)
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            step(data, pool, h)
            torch.npu.synchronize()
            samples[index].append((time.perf_counter_ns() - start) / 1e6)
    memory = [peak_allocated(data, pool, h, step) for _, pool, h, step in paths]
    legacy = summarize(samples[0])
    pooled = summarize(samples[1])
    result = {
        "device": torch.npu.get_device_name(0),
        "workload": args.workload,
        "batch": batch,
        "suffix_tokens_per_request": args.suffix_length,
        "key_heads": args.key_heads,
        "value_heads": args.value_heads,
        "head_dim": args.head_dim,
        "dtype": "bfloat16",
        "correctness": "bitwise output, chunk_state and full pool match",
        "scope": "state I/O only; prefix COW and state preparation are outside timed regions",
        "timing": "one call per round, synchronized before and after, alternating order",
        "legacy": legacy,
        "pool": pooled,
        "median_reduction_percent": 100
        * (1 - pooled["median_ms"] / legacy["median_ms"]),
        "legacy_memory": memory[0],
        "pool_memory": memory[1],
        "incremental_peak_saved_bytes": memory[0]["incremental_peak_bytes"]
        - memory[1]["incremental_peak_bytes"],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        f"{args.workload} B={batch} L={args.suffix_length}: "
        f"legacy={legacy['median_ms']:.4f} ms, pool={pooled['median_ms']:.4f} ms, "
        f"reduction={result['median_reduction_percent']:.2f}%, "
        f"peak saved={result['incremental_peak_saved_bytes'] / 1024**2:.2f} MiB"
    )
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
