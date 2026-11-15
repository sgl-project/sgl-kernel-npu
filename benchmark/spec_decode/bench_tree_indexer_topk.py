"""Standalone correctness/latency check for the AscendC tree indexer TopK.

CPU algorithm check (no torch/NPU dependency):
    python3 benchmark/spec_decode/bench_tree_indexer_topk.py --cpu-only
NPU correctness + latency; optionally compare the working Triton bisection:
    python3 benchmark/spec_decode/bench_tree_indexer_topk.py --compare-triton

This checks selection from scores, not the upstream gather/score kernels or
end-to-end generation. It never invokes the failing tl.topk implementation.
"""

import argparse
import importlib
import math
import random
import time

TILE = 1024
NEG_INF = float("-inf")


def reference(row, k):
    valid = [i for i, score in enumerate(row) if score > NEG_INF]
    chosen = sorted(valid, key=lambda i: (-row[i], i))[:k]
    return sorted(chosen) + [-1] * (k - len(chosen))


def streaming_model(row, k):
    """Model the native threshold/compaction algorithm, not native execution."""
    keep = (k + 31) // 32 * 32
    clean = [max(NEG_INF, score) for score in row]
    winners = []
    for begin in range(0, len(clean), TILE):
        tile = clean[begin : begin + TILE]
        tile += [NEG_INF] * (TILE - len(tile))
        winners = sorted(winners + sorted(tile, reverse=True)[:keep], reverse=True)[
            :keep
        ]
    threshold = winners[k - 1]
    greater = sum(score > threshold for score in clean)
    ties_left = k - greater
    chosen = []
    for i, score in enumerate(clean):
        if score > threshold:
            chosen.append(i)
        elif score == threshold and score > NEG_INF and ties_left:
            chosen.append(i)
            ties_left -= 1
    return chosen + [-1] * (k - len(chosen))


def cases():
    rng = random.Random(20261111)
    for slots in (1, 31, 33, 511, 512, 1023, 1024, 1025, 2049, 4096, 8192):
        for k in (1, 31, 32, 33, 511, 512, 1024):
            rows = [
                [rng.uniform(-10, 10) for _ in range(slots)],
                [float(rng.randrange(-3, 4)) for _ in range(slots)],
                [NEG_INF] * slots,
                [0.0] * slots,
                [NEG_INF if i % 17 else float(i % 7) for i in range(slots)],
                # Cross-tile ties, signed zeros, nonfinite and extreme scores.
                [
                    (float("nan"), NEG_INF, float("inf"), -0.0, 0.0, 1e30, -1e30)[i % 7]
                    for i in range(slots)
                ],
            ]
            yield f"slots={slots},k={k}", rows, k
    # More rows than the A3 vector cores, to exercise core reuse.
    yield (
        "many_rows",
        [[float(rng.randrange(-16, 17)) for _ in range(1057)] for _ in range(129)],
        512,
    )


def check_cpu():
    count = 0
    for name, rows, k in cases():
        for row in rows:
            assert streaming_model(row, k) == reference(row, k), name
            count += 1
    print(f"CPU algorithm/reference: {count} rows passed (not a native kernel test)")


def check_npu(torch, op, device):
    count = 0
    for name, rows, k in cases():
        scores = torch.tensor(rows, dtype=torch.float32)
        # Compute the oracle after rounding to FP32, as the device sees it.
        expected = torch.tensor(
            [reference(row, k) for row in scores.tolist()], dtype=torch.int32
        ).unsqueeze(1)
        actual = op(scores.to(device), k).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
        count += len(rows)
    empty = op(torch.empty((0, 1024), dtype=torch.float32, device=device), 512)
    assert tuple(empty.shape) == (0, 1, 512)
    print(f"NPU exact index/order/padding: {count} rows + empty batch passed")


def measure(torch, call, warmup, repeats):
    torch.npu.synchronize()
    start = time.perf_counter()
    first = call()
    torch.npu.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start = time.perf_counter()
    start_event.record()
    for _ in range(repeats):
        call()
    end_event.record()
    torch.npu.synchronize()
    wall_us = (time.perf_counter() - start) * 1e6 / repeats
    stream_us = start_event.elapsed_time(end_event) * 1000 / repeats
    # The stream interval can include launch gaps; use profiling for pure kernel time.
    return first, cold_ms, stream_us, wall_us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--compare-triton", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, default=13)
    parser.add_argument(
        "--slots", type=int, nargs="+", default=[1024, 2048, 4096, 8192]
    )
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.rows < 1 or not 1 <= args.k <= 1024 or min(args.slots) < 1:
        parser.error("rows/slots must be positive; k must be in [1, 1024]")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    if args.compare_triton and any(s < 512 or s & (s - 1) for s in args.slots):
        parser.error("Triton comparison requires power-of-two slots >= 512")
    check_cpu()
    if args.cpu_only:
        return

    import torch

    # Package initialization loads torch_npu and registers the compiled library.
    importlib.import_module("sgl_kernel_npu")
    torch.npu.set_device(args.device)
    op = getattr(torch.ops.npu, "tree_indexer_topk", None)
    if op is None:
        raise RuntimeError(
            "tree_indexer_topk is missing: rebuild/install the A2/A3 wheel"
        )
    check_npu(torch, op, args.device)
    module = None
    if args.compare_triton:
        module = importlib.import_module(
            "sgl_kernel_npu.tree.quant_lightning_indexer_tree"
        )
    print("impl rows slots k first_call_ms warm_stream_us warm_wall_us")
    generator = torch.Generator().manual_seed(20261111)
    for slots in args.slots:
        cpu = torch.randn((args.rows, slots), generator=generator)
        cpu[:, ::17] = NEG_INF
        scores = cpu.to(args.device)
        expected = torch.tensor(
            [reference(row, args.k) for row in cpu.tolist()], dtype=torch.int32
        ).unsqueeze(1)
        calls = {"ascendc": lambda scores=scores: op(scores, args.k)}
        if module is not None:

            def bisection(scores=scores, slots=slots):
                out = torch.empty(
                    (args.rows, 1, args.k), dtype=torch.int32, device=args.device
                )
                module._select_topk_kernel[(args.rows,)](
                    scores,
                    out,
                    args.k,
                    SLOTS=slots,
                    TILE=min(2048, slots),
                    FILL=module._MAX_TOPK,
                    BISECT_STEPS=32,
                    num_warps=module._NUM_WARPS,
                    num_stages=module._NUM_STAGES,
                )
                return out

            calls["triton_bisection"] = bisection
        for name, call in calls.items():
            actual, cold_ms, stream_us, wall_us = measure(
                torch, call, args.warmup, args.repeats
            )
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            assert math.isfinite(stream_us) and stream_us >= 0
            print(
                f"{name} {args.rows} {slots} {args.k} "
                f"{cold_ms:.3f} {stream_us:.3f} {wall_us:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
