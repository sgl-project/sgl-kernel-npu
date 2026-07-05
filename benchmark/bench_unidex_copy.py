#!/usr/bin/env python3
"""Benchmark and validate unidex_copy across D2D/H2D/D2H directions.

Compares unidex_copy against a pure-PyTorch index_select + index_copy
baseline.  hit_rate controls valid_mask density; index modes control
address pattern density.

Usage:
    # D2D with default settings
    python benchmark/bench_unidex_copy.py

    # Sweep hit rates
    python benchmark/bench_unidex_copy.py --hit-rates 0.0 0.25 0.5 0.75 1.0

    # Realistic MLA workload
    python benchmark/bench_unidex_copy.py --batch-size 8 --topk 2048 \\
        --token-bytes 1152 --head-num 1 --head-dim 576 --dtype float16
"""

import argparse
import itertools
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch_npu  # noqa: F401

import sgl_kernel_npu  # noqa: F401

DEVICE = "npu"
DTYPE_MAP = {
    "uint8": torch.uint8,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class BenchmarkCase:
    direction: str
    src: torch.Tensor
    dst: torch.Tensor
    dst_before: torch.Tensor
    src_flat_cpu: torch.Tensor
    dst_before_flat_cpu: torch.Tensor
    src_index: torch.Tensor
    dst_index: torch.Tensor
    valid_mask: torch.Tensor
    hit_count: int
    src_rows: int
    dst_rows: int
    block_bytes: int
    block_elems: int
    block_dim: int


def synchronize():
    torch.npu.synchronize()


def make_flat_rows(rows, block_elems, dtype, offset):
    row_values = (torch.arange(rows, dtype=torch.int32) % 251).reshape(rows, 1)
    col_values = (torch.arange(block_elems, dtype=torch.int32) % 251).reshape(1, block_elems)
    data = ((row_values + col_values + offset) % 251).contiguous()
    return data.to(dtype=dtype).contiguous()


def make_index(mode, rows, max_copy, generator, unique=False):
    if mode == "arange":
        return torch.arange(max_copy, dtype=torch.long) % rows
    if mode == "random":
        if unique:
            if rows < max_copy:
                raise ValueError(
                    f"random dst_index requires dst_rows >= max_copy, got dst_rows={rows}, max_copy={max_copy}"
                )
            return torch.randperm(rows, generator=generator, dtype=torch.long)[:max_copy]
        return torch.randint(0, rows, (max_copy,), generator=generator, dtype=torch.long)
    raise ValueError(f"Unsupported index mode: {mode}")


def make_valid_mask(max_copy, hit_rate, generator):
    hit_count = int(round(max_copy * hit_rate))
    valid_cpu = torch.zeros(max_copy, dtype=torch.bool)
    if hit_count == max_copy:
        valid_cpu.fill_(True)
    elif hit_count > 0:
        perm = torch.randperm(max_copy, generator=generator)
        valid_cpu[perm[:hit_count]] = True
    return valid_cpu, hit_count


def copy_to_location(flat_cpu, shape, direction, role):
    tensor_cpu = flat_cpu.view(shape).contiguous()
    needs_cpu = (direction == "h2d" and role == "src") or (direction == "d2h" and role == "dst")
    if needs_cpu:
        return tensor_cpu, None
    return tensor_cpu.to(DEVICE).contiguous(), None


def make_case(args, seed_offset=0, direction=None):
    direction = direction or getattr(args, "direction", "d2d")
    max_copy = args.batch_size * args.topk
    if max_copy <= 0:
        raise ValueError("batch_size * topk must be positive")

    dtype = DTYPE_MAP[args.dtype]
    elem_size = torch.empty((), dtype=dtype).element_size()
    if args.token_bytes % elem_size != 0:
        raise ValueError(f"token_bytes ({args.token_bytes}) must be divisible by dtype element size ({elem_size})")
    if args.token_bytes % 32 != 0:
        raise ValueError("token_bytes must be a multiple of 32 for aligned DataCopy")
    if args.token_bytes != args.head_num * args.head_dim * elem_size:
        raise ValueError(
            "token_bytes must equal head_num * head_dim * element_size. "
            f"Got token_bytes={args.token_bytes}, head_num={args.head_num}, "
            f"head_dim={args.head_dim}, element_size={elem_size}."
        )
    if not 0.0 <= args.hit_rate <= 1.0:
        raise ValueError("hit_rate must be in [0, 1]")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + seed_offset)

    src_rows = args.src_rows
    block_elems = args.head_num * args.head_dim
    dst_rows = max(args.dst_rows, max_copy) if args.dst_rows > 0 else max_copy

    if dst_rows < max_copy:
        raise ValueError(f"dst_rows ({dst_rows}) must be >= max_copy ({max_copy})")

    src_index_cpu = make_index(args.src_index_mode, src_rows, max_copy, generator)
    dst_index_cpu = make_index(args.dst_index_mode, dst_rows, max_copy, generator, unique=True)
    valid_cpu, hit_count = make_valid_mask(max_copy, args.hit_rate, generator)

    src_flat_cpu = make_flat_rows(src_rows, block_elems, dtype, offset=0)
    dst_before_flat_cpu = make_flat_rows(dst_rows, block_elems, dtype, offset=17)

    src_shape = (src_rows, block_elems)
    dst_shape = (dst_rows, block_elems)

    src, _ = copy_to_location(src_flat_cpu, src_shape, direction, "src")
    dst, _ = copy_to_location(dst_before_flat_cpu, dst_shape, direction, "dst")
    dst_before = dst.clone()

    return BenchmarkCase(
        direction=direction,
        src=src,
        dst=dst,
        dst_before=dst_before,
        src_flat_cpu=src_flat_cpu,
        dst_before_flat_cpu=dst_before_flat_cpu,
        src_index=src_index_cpu.to(DEVICE).contiguous(),
        dst_index=dst_index_cpu.to(DEVICE).contiguous(),
        valid_mask=valid_cpu.to(DEVICE).contiguous(),
        hit_count=hit_count,
        src_rows=src_rows,
        dst_rows=dst_rows,
        block_bytes=args.token_bytes,
        block_elems=block_elems,
        block_dim=args.block_dim,
    )


def unidex_copy_kernel(case):
    torch.ops.npu.unidex_copy(
        case.src,
        case.dst,
        case.src_index,
        case.dst_index,
        case.valid_mask,
        case.src_rows,
        case.dst_rows,
        case.block_bytes,
        case.src_index.numel(),
        case.block_dim,
    )


def torch_index_copy(case):
    valid_pos = torch.nonzero(case.valid_mask, as_tuple=False).flatten()
    src_rows = case.src_index.index_select(0, valid_pos)
    dst_rows = case.dst_index.index_select(0, valid_pos)
    tmp = case.src.index_select(0, src_rows)
    case.dst.index_copy_(0, dst_rows, tmp)


def run_copy(args, case, baseline, sync=False):
    if baseline == "unidex":
        unidex_copy_kernel(case)
    elif baseline == "torch_index_copy":
        torch_index_copy(case)
    if sync:
        synchronize()


def check_correctness(case, baseline="unidex"):
    synchronize()
    src_index_cpu = case.src_index.cpu()
    dst_index_cpu = case.dst_index.cpu()
    valid_cpu = case.valid_mask.cpu()
    expected = case.dst_before_flat_cpu.clone()
    expected[dst_index_cpu[valid_cpu]] = case.src_flat_cpu[src_index_cpu[valid_cpu]]

    actual = case.dst.cpu()
    if torch.equal(actual, expected):
        return

    diff = (actual != expected).nonzero(as_tuple=False)
    first = diff[0]
    row = int(first[0].item())
    col = int(first[1].item())
    raise AssertionError(
        f"correctness failed: baseline={baseline}, direction={case.direction}, "
        f"mismatch_count={int(diff.shape[0])}, first=({row}, {col}), "
        f"actual={actual[row, col].item()}, expected={expected[row, col].item()}"
    )


def time_average_ms(fn, warmup, perf_iters):
    for _ in range(warmup):
        fn()
    synchronize()

    start = time.perf_counter()
    for _ in range(perf_iters):
        fn()
    synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / perf_iters


def run_benchmark_direction(args, direction, baseline):
    if args.accuracy_iters > 0:
        for i in range(args.accuracy_iters):
            case = make_case(args, seed_offset=i, direction=direction)
            dst_before = case.dst.clone()
            run_copy(args, case, baseline, sync=False)
            check_correctness(case, baseline)

    case = make_case(args, direction=direction)
    latency_ms = time_average_ms(
        lambda: run_copy(args, case, baseline, sync=False),
        args.warmup,
        args.perf_iters,
    )

    payload_bytes = case.hit_count * case.block_bytes
    memory_bytes = payload_bytes * 2
    mean_s = latency_ms / 1000.0
    payload_gbs = payload_bytes / mean_s / 1e9 if mean_s > 0 else float("inf")
    memory_gbs = memory_bytes / mean_s / 1e9 if mean_s > 0 else float("inf")
    return case, latency_ms, payload_gbs, memory_gbs


def print_case_result(case, args, baseline, latency_ms, payload_gbs, memory_gbs):
    print()
    print(f"baseline={baseline}, direction={case.direction}")
    print(f"batch_size={args.batch_size}, topk={args.topk}, max_copy={args.batch_size * args.topk}, "
          f"src_rows={case.src_rows}, dst_rows={case.dst_rows}, "
          f"src_index_mode={args.src_index_mode}, dst_index_mode={args.dst_index_mode}, "
          f"hit_rate={args.hit_rate:.4f}, hit_count={case.hit_count}")
    print(f"dtype={args.dtype}, head_num={args.head_num}, head_dim={args.head_dim}, "
          f"token_bytes={case.block_bytes}, block_dim={case.block_dim}")
    print(f"warmup={args.warmup}, perf_iters={args.perf_iters}, accuracy_iters={args.accuracy_iters}")
    print(f"latency_ms={latency_ms:.6f}")
    print(f"payload_bandwidth={payload_gbs:.3f} GB/s")
    print(f"read_write_bandwidth={memory_gbs:.3f} GB/s")


def validate_args(args):
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.perf_iters <= 0:
        raise ValueError("perf_iters must be positive")
    if args.accuracy_iters < 0:
        raise ValueError("accuracy_iters must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    if args.src_rows <= 0:
        raise ValueError("src_rows must be positive")
    if args.dst_rows < 0:
        raise ValueError("dst_rows must be non-negative")
    if args.head_num <= 0:
        raise ValueError("head_num must be positive")
    dtype = DTYPE_MAP[args.dtype]
    elem_size = torch.empty((), dtype=dtype).element_size()
    if args.head_dim == 0:
        if args.token_bytes % (args.head_num * elem_size) != 0:
            raise ValueError("token_bytes must be divisible by head_num * element_size when head_dim is auto")
        args.head_dim = args.token_bytes // (args.head_num * elem_size)
    elif args.head_dim < 0:
        raise ValueError("head_dim must be non-negative")
    if not 0.0 <= args.hit_rate <= 1.0:
        raise ValueError("hit_rate must be in [0, 1]")
    if args.hit_rates is not None:
        for hit_rate in args.hit_rates:
            if not 0.0 <= hit_rate <= 1.0:
                raise ValueError("all hit_rates must be in [0, 1]")

    seen = set()
    unique_directions = []
    for direction in args.directions:
        if direction not in seen:
            seen.add(direction)
            unique_directions.append(direction)
    args.directions = unique_directions

    args.hit_rates = args.hit_rates or [args.hit_rate]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and validate unidex_copy across D2D/H2D/D2H directions."
    )
    parser.add_argument("--directions", nargs="+", choices=("d2d", "h2d", "d2h"), default=["d2d"])
    parser.add_argument("--baselines", nargs="+", choices=("unidex", "torch_index_copy"), default=["unidex"])
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--src-rows", type=int, default=128000)
    parser.add_argument("--dst-rows", type=int, default=0, help="0 means infer from max_copy")
    parser.add_argument("--src-index-mode", choices=("random", "arange"), default="random")
    parser.add_argument("--dst-index-mode", choices=("arange", "random"), default="arange")
    parser.add_argument("--hit-rate", type=float, default=0.5)
    parser.add_argument("--hit-rates", nargs="+", type=float, default=None)
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP.keys()), default="float16")
    parser.add_argument("--head-num", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=0,
                        help="0 means infer from token_bytes/dtype/head_num")
    parser.add_argument("--token-bytes", type=int, default=1152)
    parser.add_argument("--block-dim", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--perf-iters", type=int, default=100)
    parser.add_argument("--accuracy-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()
    validate_args(args)

    cases = itertools.product(
        args.directions, args.baselines, args.hit_rates,
    )
    for idx, (direction, baseline, hit_rate) in enumerate(cases):
        args.hit_rate = hit_rate
        case, latency_ms, payload_gbs, memory_gbs = run_benchmark_direction(args, direction, baseline)
        print_case_result(case, args, baseline, latency_ms, payload_gbs, memory_gbs)


if __name__ == "__main__":
    main()
