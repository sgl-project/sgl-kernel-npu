#!/usr/bin/env python3
"""Compare slot_map_lookup with broadcast + any + argmax.

Builds equivalent inputs for both approaches, verifies correctness,
then measures per-iteration latency with NPU device events and
reports the operator-only speedup.

Usage:
    python benchmark/bench_slot_vs_argmax.py
    python benchmark/bench_slot_vs_argmax.py --bs 8 --topk-len 4096 --hit-ratio 0.3
"""

import argparse
import statistics
import sys

import torch
import torch_npu  # noqa: F401

import sgl_kernel_npu  # noqa: F401
from sgl_kernel_npu.mem_cache import slot_map_lookup

DEVICE = "npu"


def log(message):
    print(message, flush=True)


def synchronize():
    torch.npu.synchronize()


def build_argmax_reference(topk_indices_i64, device_kv_indices_i64):
    token_match_matrix = topk_indices_i64.unsqueeze(-1) == device_kv_indices_i64.unsqueeze(1)
    token_on_device = token_match_matrix.any(dim=-1)
    device_token_pos = token_match_matrix.int().argmax(dim=-1)
    return token_on_device, device_token_pos


def run_any_argmax(token_match_matrix, token_match_matrix_i32):
    token_on_device = token_match_matrix.any(dim=-1)
    device_token_pos = token_match_matrix_i32.argmax(dim=-1)
    return token_on_device, device_token_pos


def build_equivalent_inputs(size, bs, max_context_len, topk_len, device_len, hit_ratio, seed):
    if size < bs:
        raise ValueError("size must be >= bs")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    req_indices_cpu = torch.randperm(size, generator=generator)[:bs].to(torch.int32)
    slot_map_cpu = torch.full((size, max_context_len), -1, dtype=torch.int32)
    device_kv_cpu = torch.empty((bs, device_len), dtype=torch.long)
    topk_cpu = torch.empty((bs, topk_len), dtype=torch.long)

    hit_count = max(0, min(topk_len, int(round(topk_len * hit_ratio))))
    miss_count = topk_len - hit_count
    valid_device_len = min(device_len, max_context_len)

    for b in range(bs):
        req_id = int(req_indices_cpu[b].item())
        token_perm = torch.randperm(max_context_len, generator=generator, dtype=torch.long)
        device_tokens = token_perm[:valid_device_len]
        miss_tokens = token_perm[valid_device_len:]

        device_kv_cpu[b, :valid_device_len] = device_tokens
        slot_map_cpu[req_id, device_tokens] = torch.arange(valid_device_len, dtype=torch.int32)
        if device_len > valid_device_len:
            device_kv_cpu[b, valid_device_len:] = torch.arange(
                max_context_len,
                max_context_len + device_len - valid_device_len,
                dtype=torch.long,
            )

        pieces = []
        if hit_count > 0:
            hit_slots = torch.randint(
                low=0,
                high=valid_device_len,
                size=(hit_count,),
                dtype=torch.long,
                generator=generator,
            )
            pieces.append(device_tokens[hit_slots])
        if miss_count > 0:
            if miss_tokens.numel() > 0:
                miss_slots = torch.randint(
                    low=0,
                    high=int(miss_tokens.numel()),
                    size=(miss_count,),
                    dtype=torch.long,
                    generator=generator,
                )
                pieces.append(miss_tokens[miss_slots])
            else:
                pieces.append(torch.full((miss_count,), -1, dtype=torch.long))

        row = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        topk_cpu[b] = row[torch.randperm(topk_len, generator=generator)]

    return (
        slot_map_cpu.to(device=DEVICE).contiguous(),
        req_indices_cpu.to(device=DEVICE).contiguous(),
        topk_cpu.to(device=DEVICE).contiguous(),
        device_kv_cpu.to(device=DEVICE).contiguous(),
    )


def normalize_slot_outputs(slot_mask, slot_pos_i32):
    slot_mask = slot_mask.to(torch.bool)
    slot_pos_long = slot_pos_i32.to(torch.long)
    return slot_mask, torch.where(slot_mask, slot_pos_long, torch.zeros_like(slot_pos_long))


def assert_equal_pair(name, actual_mask, actual_pos, expected_mask, expected_pos):
    if torch.equal(actual_mask, expected_mask) and torch.equal(actual_pos, expected_pos):
        return

    diff_mask = (actual_mask != expected_mask).nonzero(as_tuple=False)
    diff_pos = (actual_pos != expected_pos).nonzero(as_tuple=False)
    log(f"{name} mismatch")
    log(f"mask mismatch count: {diff_mask.shape[0]}")
    print(diff_mask[:20], flush=True)
    if diff_mask.numel() > 0:
        first = diff_mask[0]
        b = int(first[0].item())
        k = int(first[1].item())
        log(f"first mask mismatch at [{b}, {k}]: actual={bool(actual_mask[b, k].item())}, "
            f"expected={bool(expected_mask[b, k].item())}")
    log(f"pos mismatch count: {diff_pos.shape[0]}")
    print(diff_pos[:20], flush=True)
    if diff_pos.numel() > 0:
        first = diff_pos[0]
        b = int(first[0].item())
        k = int(first[1].item())
        log(f"first pos mismatch at [{b}, {k}]: actual={int(actual_pos[b, k].item())}, "
            f"expected={int(expected_pos[b, k].item())}")
    raise SystemExit(1)


def build_case_data(args, seed):
    slot_map, req_indices_i32, topk_indices_i64, device_kv_indices_i64 = build_equivalent_inputs(
        size=args.size,
        bs=args.bs,
        max_context_len=args.max_context_len,
        topk_len=args.topk_len,
        device_len=args.device_len,
        hit_ratio=args.hit_ratio,
        seed=seed,
    )
    topk_indices_i32 = topk_indices_i64.to(torch.int32)
    expected_mask, expected_pos = build_argmax_reference(topk_indices_i64, device_kv_indices_i64)
    synchronize()
    return slot_map, req_indices_i32, topk_indices_i64, topk_indices_i32, device_kv_indices_i64, expected_mask, expected_pos


def check_slot_lookup(args, block_dim, case_data, name):
    slot_map, req_indices_i32, _topk_indices_i64, topk_indices_i32, _device_kv_indices_i64, expected_mask, expected_pos = case_data
    actual_token, actual_pos = slot_map_lookup(
        slot_map, req_indices_i32, topk_indices_i32, block_dim=block_dim,
    )
    synchronize()
    slot_mask, slot_pos = normalize_slot_outputs(actual_token, actual_pos)
    assert_equal_pair(name, slot_mask, slot_pos, expected_mask, expected_pos)


def time_samples_ms(fn, warmup, iters, repeat):
    result = None
    for _ in range(warmup):
        result = fn()
    synchronize()

    samples = []
    for _ in range(repeat):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            result = fn()
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event) / iters)
    del result
    return samples


def summarize_samples(samples):
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean, median, min(samples), max(samples), std


def log_samples(label, samples):
    mean, median, min_value, max_value, std = summarize_samples(samples)
    log(f"{label}: mean={mean:.3f} ms, median={median:.3f} ms, "
        f"min={min_value:.3f} ms, max={max_value:.3f} ms, std={std:.3f} ms")


def log_argmax_working_set(bs, topk_len, device_len):
    match_elements = bs * topk_len * device_len
    bool_mib = match_elements / (1024 ** 2)
    int32_mib = match_elements * 4 / (1024 ** 2)
    log(f"any+argmax intermediate tensors: shape=[{bs}, {topk_len}, {device_len}], "
        f"bool match={bool_mib:.1f} MiB, int32 copy={int32_mib:.1f} MiB, "
        f"combined={bool_mib + int32_mib:.1f} MiB")


def accuracy_case(args, case_idx, block_dim):
    log(f"\n[ACCURACY] case={case_idx}, size={args.size}, bs={args.bs}, topk={args.topk_len}, "
        f"device_len={args.device_len}, max_context_len={args.max_context_len}, "
        f"hit_ratio={args.hit_ratio:.2f}, block_dim={block_dim}")
    case_data = build_case_data(args, args.seed + case_idx)
    check_slot_lookup(args, block_dim, case_data, "slot_map_lookup vs any+argmax")


def benchmark_case(args, block_dim, case_data):
    log(f"\n[PERF] size={args.size}, bs={args.bs}, topk={args.topk_len}, "
        f"device_len={args.device_len}, max_context_len={args.max_context_len}, "
        f"hit_ratio={args.hit_ratio:.2f}, block_dim={block_dim}, "
        f"perf_repeat={args.perf_repeat}, iters={args.iters}, timing=npu_event")
    slot_map, req_indices_i32, topk_indices_i64, topk_indices_i32, device_kv_indices_i64, _expected_mask, _expected_pos = case_data

    slot_mask_u8, slot_pos_i32 = slot_map_lookup(
        slot_map, req_indices_i32, topk_indices_i32, block_dim=block_dim,
    )
    synchronize()

    token_match_matrix = topk_indices_i64.unsqueeze(-1) == device_kv_indices_i64.unsqueeze(1)
    token_match_matrix_i32 = token_match_matrix.int()
    synchronize()

    ref_samples = time_samples_ms(
        lambda: run_any_argmax(token_match_matrix, token_match_matrix_i32),
        args.warmup, args.iters, args.perf_repeat,
    )
    del token_match_matrix, token_match_matrix_i32
    synchronize()

    slot_samples = time_samples_ms(
        lambda: slot_map_lookup(slot_map, req_indices_i32, topk_indices_i32, block_dim=block_dim),
        args.warmup, args.iters, args.perf_repeat,
    )

    ref_mean, ref_median, _, _, _ = summarize_samples(ref_samples)
    slot_mean, slot_median, _, _, _ = summarize_samples(slot_samples)
    log_argmax_working_set(args.bs, args.topk_len, args.device_len)
    log_samples("any+argmax operators only (prebuilt match matrices)", ref_samples)
    log_samples("slot_map_lookup kernel only (preallocated outputs)", slot_samples)
    log(f"operator-only speedup: mean={ref_mean / slot_mean:.2f}x, median={ref_median / slot_median:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare slot_map_lookup with broadcast+any+argmax.")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--topk-len", type=int, default=2048)
    parser.add_argument("--device-len", type=int, default=2048,
                        help="Physical length of device KV list used by any+argmax.")
    parser.add_argument("--max-context-len", type=int, default=128000)
    parser.add_argument("--hit-ratio", type=float, default=0.5)
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--accuracy-repeat", type=int, default=1)
    parser.add_argument("--perf-repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260514)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.hit_ratio <= 1.0:
        raise SystemExit("--hit-ratio must be in [0, 1]")
    if args.block_dim <= 0:
        raise SystemExit("--block-dim must be positive")

    log("slot_map_lookup vs any+argmax benchmark started.")
    for case_idx in range(args.accuracy_repeat):
        accuracy_case(args, case_idx, args.block_dim)

    perf_case_data = build_case_data(args, args.seed)
    benchmark_case(args, args.block_dim, perf_case_data)
    log("\nslot_map_lookup vs any+argmax benchmark passed.")


if __name__ == "__main__":
    main()
