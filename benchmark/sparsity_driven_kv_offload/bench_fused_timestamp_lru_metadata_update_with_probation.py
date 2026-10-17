"""Benchmark the timestamp-LRU selection and metadata-write kernels on NPU.

Each kernel is timed independently with NPU events. Its mutated metadata is
restored before the start event, so reset copies are excluded from latency.

Examples:
    # Default case: batch size 32 with an exact 50% hit rate.
    python benchmark/sparsity_driven_kv_offload/bench_fused_timestamp_lru_metadata_update_with_probation.py

    # One batch size and several hit rates.
    python benchmark/sparsity_driven_kv_offload/bench_fused_timestamp_lru_metadata_update_with_probation.py \
        --batch-size 32 --hit-rates 0.0 0.5 1.0

    # Sweep batch sizes with a manually selected AIV block count.
    python benchmark/sparsity_driven_kv_offload/bench_fused_timestamp_lru_metadata_update_with_probation.py \
        --batch-sizes 1 8 16 32 --block-dim 8 --iters 100
"""

import argparse
import math
import statistics
from dataclasses import dataclass

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    fused_timestamp_lru_metadata_update_with_probation,
    parallel_lru_metadata_write,
)


DEVICE = "npu"
TOPK = 2048
CACHE_CAPACITY = 4096
DEFAULT_STAMP_MAX = (1 << 24) - 1


@dataclass
class BenchmarkCase:
    batch_size: int
    hit_count: int
    miss_count: int
    max_context_len: int
    stamp_max: int
    probation_age: int
    block_dim: int
    metadata_block_dim: int
    slot_map: torch.Tensor
    req_indices: torch.Tensor
    topk_indices: torch.Tensor
    device_token_pos: torch.Tensor
    hit_position_mask: torch.Tensor
    lru_slots: torch.Tensor
    lru_stamps: torch.Tensor
    slot_tokens: torch.Tensor
    initial_slot_map: torch.Tensor
    initial_lru_slots: torch.Tensor
    initial_lru_stamps: torch.Tensor
    initial_slot_tokens: torch.Tensor
    topk_indices_cpu: torch.Tensor


def synchronize():
    torch.npu.synchronize()


def make_case(
    batch_size,
    hit_rate,
    max_context_len,
    stamp_max,
    probation_age,
    block_dim,
    metadata_block_dim,
    seed,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError("hit_rate must be in [0, 1]")
    if max_context_len % 32 != 0:
        raise ValueError("max_context_len must be a multiple of 32")
    if not CACHE_CAPACITY + 1 <= stamp_max <= DEFAULT_STAMP_MAX:
        raise ValueError(
            "stamp_max must be in "
            f"[{CACHE_CAPACITY + 1}, {DEFAULT_STAMP_MAX}] so this benchmark "
            "can construct unique LRU ages"
        )
    if not 0 <= probation_age <= stamp_max:
        raise ValueError("probation_age must be in [0, stamp_max]")
    if block_dim < 0:
        raise ValueError("block_dim must be non-negative")
    if metadata_block_dim < 0:
        raise ValueError("metadata_block_dim must be non-negative")

    hit_count = max(0, min(TOPK, int(round(TOPK * hit_rate))))
    miss_count = TOPK - hit_count
    if max_context_len < CACHE_CAPACITY + miss_count:
        raise ValueError(
            "max_context_len must be at least cache_capacity + miss_count; "
            f"got {max_context_len} < {CACHE_CAPACITY + miss_count}"
        )

    rows = batch_size + 1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    req_indices_cpu = torch.arange(1, rows, dtype=torch.int32)
    resident_tokens = torch.arange(CACHE_CAPACITY, dtype=torch.int32)
    hit_tokens = resident_tokens[:hit_count]
    miss_tokens = torch.arange(
        CACHE_CAPACITY,
        CACHE_CAPACITY + miss_count,
        dtype=torch.int32,
    )
    candidate_tokens = torch.cat((hit_tokens, miss_tokens))

    topk_indices_cpu = torch.empty((batch_size, TOPK), dtype=torch.int32)
    for batch_idx in range(batch_size):
        permutation = torch.randperm(TOPK, generator=generator)
        topk_indices_cpu[batch_idx] = candidate_tokens[permutation]
    device_token_pos_cpu = torch.where(
        topk_indices_cpu < CACHE_CAPACITY,
        topk_indices_cpu,
        torch.full_like(topk_indices_cpu, -1),
    )
    hit_position_mask_cpu = torch.zeros(
        (batch_size, CACHE_CAPACITY), dtype=torch.int32
    )
    hit_position_mask_cpu[:, :hit_count] = 1

    slot_map_cpu = torch.full(
        (rows, max_context_len), -1, dtype=torch.int32
    )
    slot_map_cpu[1:, :CACHE_CAPACITY] = resident_tokens

    lru_slots_cpu = resident_tokens.expand(rows, CACHE_CAPACITY).clone()
    base_stamps = torch.arange(
        CACHE_CAPACITY, 0, -1, dtype=torch.int32
    )
    lru_stamps_cpu = base_stamps.expand(rows, CACHE_CAPACITY).clone()

    slot_tokens_cpu = torch.full(
        (rows, CACHE_CAPACITY), -1, dtype=torch.int32
    )
    slot_tokens_cpu[1:] = resident_tokens

    slot_map = slot_map_cpu.to(DEVICE).contiguous()
    lru_slots = lru_slots_cpu.to(DEVICE).contiguous()
    lru_stamps = lru_stamps_cpu.to(DEVICE).contiguous()
    slot_tokens = slot_tokens_cpu.to(DEVICE).contiguous()

    return BenchmarkCase(
        batch_size=batch_size,
        hit_count=hit_count,
        miss_count=miss_count,
        max_context_len=max_context_len,
        stamp_max=stamp_max,
        probation_age=probation_age,
        block_dim=block_dim,
        metadata_block_dim=metadata_block_dim,
        slot_map=slot_map,
        req_indices=req_indices_cpu.to(DEVICE).contiguous(),
        topk_indices=topk_indices_cpu.to(DEVICE).contiguous(),
        device_token_pos=device_token_pos_cpu.to(DEVICE).contiguous(),
        hit_position_mask=hit_position_mask_cpu.to(DEVICE).contiguous(),
        lru_slots=lru_slots,
        lru_stamps=lru_stamps,
        slot_tokens=slot_tokens,
        initial_slot_map=slot_map.clone(),
        initial_lru_slots=lru_slots.clone(),
        initial_lru_stamps=lru_stamps.clone(),
        initial_slot_tokens=slot_tokens.clone(),
        topk_indices_cpu=topk_indices_cpu,
    )


def restore_case(case):
    case.slot_map.copy_(case.initial_slot_map)
    case.lru_slots.copy_(case.initial_lru_slots)
    case.lru_stamps.copy_(case.initial_lru_stamps)
    case.slot_tokens.copy_(case.initial_slot_tokens)


def restore_selection_state(case):
    case.lru_slots.copy_(case.initial_lru_slots)
    case.lru_stamps.copy_(case.initial_lru_stamps)


def restore_metadata_state(case):
    case.slot_map.copy_(case.initial_slot_map)
    case.slot_tokens.copy_(case.initial_slot_tokens)


def run_selection_kernel(case):
    return fused_timestamp_lru_metadata_update_with_probation(
        case.req_indices,
        case.topk_indices,
        case.device_token_pos,
        case.hit_position_mask,
        case.lru_slots,
        case.lru_stamps,
        max_context_len=case.max_context_len,
        probation_age=case.probation_age,
        stamp_max=case.stamp_max,
        block_dim=case.block_dim,
    )


def run_metadata_kernel(case, victim_slots, miss_counts):
    parallel_lru_metadata_write(
        case.slot_map,
        case.req_indices,
        case.topk_indices,
        victim_slots,
        miss_counts,
        case.slot_tokens,
        max_context_len=case.max_context_len,
        block_dim=case.metadata_block_dim,
    )


def run_operator(case):
    victim_slots, miss_counts = run_selection_kernel(case)
    run_metadata_kernel(case, victim_slots, miss_counts)
    return victim_slots


def build_expected(case):
    expected_victims = torch.full_like(case.topk_indices_cpu, -1)
    expected_slot_map = case.initial_slot_map.cpu().clone()
    expected_slot_tokens = case.initial_slot_tokens.cpu().clone()

    for batch_idx in range(case.batch_size):
        request_row = batch_idx + 1
        miss_positions = torch.nonzero(
            case.topk_indices_cpu[batch_idx] >= CACHE_CAPACITY,
            as_tuple=False,
        ).flatten()
        victims = torch.arange(
            case.hit_count,
            case.hit_count + case.miss_count,
            dtype=torch.int32,
        )
        expected_victims[batch_idx, miss_positions] = victims
        new_tokens = case.topk_indices_cpu[batch_idx, miss_positions]
        expected_slot_map[request_row, victims.long()] = -1
        expected_slot_map[request_row, new_tokens.long()] = victims
        expected_slot_tokens[request_row, victims.long()] = new_tokens

    expected_stamp_by_slot = torch.arange(
        CACHE_CAPACITY + 1, 1, -1, dtype=torch.int32
    )
    expected_stamp_by_slot[: case.hit_count] = 0
    expected_stamp_by_slot[
        case.hit_count : case.hit_count + case.miss_count
    ] = case.probation_age
    return (
        expected_victims,
        expected_slot_map,
        expected_slot_tokens,
        expected_stamp_by_slot,
    )


def assert_tensor_equal(name, actual, expected):
    if torch.equal(actual, expected):
        return
    mismatch = torch.nonzero(actual != expected, as_tuple=False)
    first = tuple(mismatch[0].tolist())
    raise AssertionError(
        f"{name} mismatch_count={mismatch.shape[0]}, first={first}, "
        f"actual={actual[first].item()}, expected={expected[first].item()}"
    )


def check_correctness(case):
    restore_case(case)
    actual_victims = run_operator(case)
    synchronize()

    (
        expected_victims,
        expected_slot_map,
        expected_slot_tokens,
        expected_stamp_by_slot,
    ) = build_expected(case)
    assert_tensor_equal("victim_slots", actual_victims.cpu(), expected_victims)
    assert_tensor_equal("slot_map", case.slot_map.cpu(), expected_slot_map)
    assert_tensor_equal(
        "device_slot_tokens", case.slot_tokens.cpu(), expected_slot_tokens
    )

    actual_slots = case.lru_slots.cpu()
    actual_stamps = case.lru_stamps.cpu()
    expected_slots = torch.arange(CACHE_CAPACITY, dtype=torch.int32)
    for request_row in range(1, case.batch_size + 1):
        row_slots = actual_slots[request_row]
        if not torch.equal(torch.sort(row_slots).values, expected_slots):
            raise AssertionError(
                f"device_lru_slots row {request_row} is not a slot permutation"
            )
        if torch.any(
            actual_stamps[request_row, :-1]
            < actual_stamps[request_row, 1:]
        ):
            raise AssertionError(
                f"device_lru_slot_stamps row {request_row} is not descending"
            )
        stamp_by_slot = torch.empty(CACHE_CAPACITY, dtype=torch.int32)
        stamp_by_slot[row_slots.long()] = actual_stamps[request_row]
        assert_tensor_equal(
            f"stamp_by_slot[{request_row}]",
            stamp_by_slot,
            expected_stamp_by_slot,
        )

    assert_tensor_equal(
        "padding_lru_slots", actual_slots[0], case.initial_lru_slots[0].cpu()
    )
    assert_tensor_equal(
        "padding_lru_stamps",
        actual_stamps[0],
        case.initial_lru_stamps[0].cpu(),
    )


def time_selection_samples_ms(case, warmup, iters):
    for _ in range(warmup):
        restore_selection_state(case)
        run_selection_kernel(case)
    synchronize()

    samples = []
    result = None
    for _ in range(iters):
        restore_selection_state(case)
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        result = run_selection_kernel(case)
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))
    del result
    return samples


def time_metadata_samples_ms(
    case, victim_slots, miss_counts, warmup, iters
):
    for _ in range(warmup):
        restore_metadata_state(case)
        run_metadata_kernel(case, victim_slots, miss_counts)
    synchronize()

    samples = []
    for _ in range(iters):
        restore_metadata_state(case)
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        run_metadata_kernel(case, victim_slots, miss_counts)
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))
    return samples


def time_kernels_ms(case, warmup, iters):
    selection_samples = time_selection_samples_ms(case, warmup, iters)

    restore_selection_state(case)
    victim_slots, miss_counts = run_selection_kernel(case)
    synchronize()
    metadata_samples = time_metadata_samples_ms(
        case, victim_slots, miss_counts, warmup, iters
    )
    return selection_samples, metadata_samples


def percentile(samples, fraction):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def print_kernel_result(name, case, samples):
    mean_ms = statistics.fmean(samples)
    median_ms = statistics.median(samples)
    std_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    p90_ms = percentile(samples, 0.90)
    throughput = case.batch_size * 1000.0 / median_ms
    print(
        f"{name}_latency_ms: mean={mean_ms:.6f}, "
        f"median={median_ms:.6f}, "
        f"p90={p90_ms:.6f}, min={min(samples):.6f}, "
        f"max={max(samples):.6f}, std={std_ms:.6f}"
    )
    print(
        f"{name}_throughput={throughput:.2f} requests/s "
        "(median latency)"
    )


def print_result(
    case,
    selection_samples,
    metadata_samples,
    warmup,
    check_enabled,
):
    actual_hit_rate = case.hit_count / TOPK

    print()
    print(
        f"batch_size={case.batch_size}, hit_rate={actual_hit_rate:.6f}, "
        f"hits_per_request={case.hit_count}, "
        f"misses_per_request={case.miss_count}, "
        f"probation_age={case.probation_age}, "
        f"selection_block_dim={case.block_dim}, "
        f"metadata_block_dim={case.metadata_block_dim}"
    )
    print(
        f"warmup={warmup}, measured_iters={len(selection_samples)}, "
        f"correctness_check={'on' if check_enabled else 'off'}, "
        "timing=npu_event, per-kernel state reset=excluded"
    )
    print_kernel_result(
        "fused_timestamp_lru_metadata_update_with_probation",
        case,
        selection_samples,
    )
    print_kernel_result(
        "parallel_lru_metadata_write",
        case,
        metadata_samples,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark timestamp-LRU selection and parallel metadata write "
            "separately with repeatable state and NPU event timing."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Single batch size used when --batch-sizes is omitted.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Optional batch-size sweep, for example: 1 8 16 32.",
    )
    parser.add_argument("--hit-rate", type=float, default=0.5)
    parser.add_argument(
        "--hit-rates",
        nargs="+",
        type=float,
        default=None,
        help="Optional hit-rate sweep, for example: 0.0 0.5 1.0.",
    )
    parser.add_argument("--max-context-len", type=int, default=8192)
    parser.add_argument("--stamp-max", type=int, default=DEFAULT_STAMP_MAX)
    parser.add_argument("--probation-age", type=int, default=0)
    parser.add_argument(
        "--block-dim",
        type=int,
        default=0,
        help=(
            "AIV count for fused_timestamp_lru_metadata_update_with_probation; "
            "0 uses "
            "min(batch_size, available AIV cores)."
        ),
    )
    parser.add_argument(
        "--metadata-block-dim",
        type=int,
        default=0,
        help=(
            "AIV count for parallel_lru_metadata_write; 0 uses all "
            "available AIV cores up to the tile count."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="NPU device ID; defaults to the current device.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the untimed correctness check before each benchmark case.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("iters must be positive")
    batch_sizes = args.batch_sizes or [args.batch_size]
    hit_rates = args.hit_rates or [args.hit_rate]
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("all batch sizes must be positive")
    if any(not 0.0 <= hit_rate <= 1.0 for hit_rate in hit_rates):
        raise ValueError("all hit rates must be in [0, 1]")
    if not 0 <= args.probation_age <= args.stamp_max:
        raise ValueError("probation_age must be in [0, stamp_max]")
    return batch_sizes, hit_rates


def main():
    args = parse_args()
    if args.device_id is not None:
        torch.npu.set_device(args.device_id)
    batch_sizes, hit_rates = validate_args(args)

    print("timestamp-LRU kernel benchmark started.", flush=True)
    for batch_size in batch_sizes:
        for hit_rate in hit_rates:
            case = make_case(
                batch_size=batch_size,
                hit_rate=hit_rate,
                max_context_len=args.max_context_len,
                stamp_max=args.stamp_max,
                probation_age=args.probation_age,
                block_dim=args.block_dim,
                metadata_block_dim=args.metadata_block_dim,
                seed=args.seed,
            )
            if not args.skip_check:
                check_correctness(case)
            selection_samples, metadata_samples = time_kernels_ms(
                case, args.warmup, args.iters
            )
            print_result(
                case,
                selection_samples,
                metadata_samples,
                args.warmup,
                not args.skip_check,
            )

    print("\ntimestamp-LRU kernel benchmark finished.", flush=True)


if __name__ == "__main__":
    main()
