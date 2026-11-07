"""Compare vendor LI and full-visible Q4 LIM at a fixed per-request union miss count.

Only the indexer operators lie between each pair of NPU events. LIM's encoded
resident map is restored on the same stream before every start event. The hit
rate denominator is Q*K; request-level refresh entries are reported separately.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401

S, B, Q, K, C, HEADS = 65536, 16, 4, 2048, 12288, 32
INVALID_SLOT = -(2**31)


def _sync():
    torch.npu.synchronize()


def _percentile(values, p):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def _summary(values):
    return {
        "count": len(values),
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "p95_us": _percentile(values, 0.95),
    }


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loaded_kernel_libraries():
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths = set()
    for line in maps.read_text().splitlines():
        name = line.rsplit(None, 1)[-1]
        if name.startswith("/") and "sgl_kernel_npu" in name and ".so" in name:
            paths.add(name)
    return [{"path": p, "sha256": _sha(p)} for p in sorted(paths) if Path(p).is_file()]


def make_inputs(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Order and FP32-to-BF16 conversion are shared by all three workers.
    query_cpu = torch.randn((B * Q, HEADS, 128), generator=generator).to(torch.bfloat16)
    key_cpu = torch.randn((B * (S // 128) * 128, 128), generator=generator).to(
        torch.bfloat16
    )
    weights_cpu = torch.rand((B * Q, HEADS), generator=generator).to(torch.bfloat16)

    def tensor_hash(tensor):
        return hashlib.sha256(
            memoryview(tensor.contiguous().view(torch.uint8).numpy())
        ).hexdigest()

    input_hashes = {
        "query": tensor_hash(query_cpu),
        "key": tensor_hash(key_cpu),
        "weights": tensor_hash(weights_cpu),
    }
    device = torch.device("npu", torch.npu.current_device())
    query = query_cpu.to(device)
    weights = weights_cpu.to(device)
    key = key_cpu.view(B * (S // 128), 128, 1, 128).to(device)
    table = torch.arange(B * (S // 128), dtype=torch.int32, device=device).view(B, -1)
    q_ends = torch.arange(Q, B * Q + 1, Q, dtype=torch.int32, device=device)
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=device)
    sources = torch.empty((B * Q, 1, K), dtype=torch.int32, device=device)
    destinations = torch.empty_like(sources)
    route_misses = torch.empty((B * Q,), dtype=torch.int32, device=device)
    union_sources = torch.empty((B, 32768), dtype=torch.int32, device=device)
    union_destinations = torch.empty_like(union_sources)
    union_counts = torch.empty((B,), dtype=torch.int32, device=device)
    pool = torch.full((B, S), INVALID_SLOT, dtype=torch.int32, device=device)
    native_args = [
        weights,
        torch.ones_like(weights, dtype=torch.float32),
        query,
        torch.ones(key.shape[:3], dtype=torch.float32, device=device),
        key,
        table,
        q_ends,
        seq_lens,
        seq_lens,
        torch.full((B,), C, dtype=torch.int32, device=device),
        torch.full((B,), -1, dtype=torch.int32, device=device),
        torch.arange(B, dtype=torch.int32, device=device),
        pool,
        sources,
        destinations,
        route_misses,
        union_sources,
        union_destinations,
        union_counts,
    ]
    return native_args, input_hashes


def vendor_li(args):
    result = torch_npu.npu_lightning_indexer(
        query=args[2],
        key=args[4],
        weights=args[0],
        actual_seq_lengths_query=args[6],
        actual_seq_lengths_key=args[7],
        block_table=args[5],
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=K,
        sparse_mode=3,
    )
    return result[0] if isinstance(result, (tuple, list)) else result


def native_lim(args):
    torch.ops.npu.fused_lightning_indexer_manage(*args, full_visible=True)


def make_resident_map(reference, misses, seed):
    reference = reference.view(B, Q, K).cpu().long()
    rng = random.Random(seed)
    maps = torch.full((B, S), INVALID_SLOT, dtype=torch.int32)
    missing_by_request = []
    union_sizes = []
    for request in range(B):
        union = sorted(set(reference[request].reshape(-1).tolist()))
        if len(union) > C or len(union) < misses:
            raise RuntimeError(
                f"Request {request}: union size {len(union)} cannot realize {misses} misses"
            )
        # Current Q4 positions are rewritten by the framework; exclude all
        # from the initial cache so refresh does not add unrelated copy entries.
        missing = {source for source in union if source >= S - Q}
        non_tail = [source for source in union if source < S - Q]
        remaining = misses - len(missing)
        if remaining < 0 or len(non_tail) < remaining:
            raise RuntimeError(f"Request {request}: cannot form the requested miss set")
        missing.update(rng.sample(non_tail, remaining))
        resident = [source for source in union if source not in missing]
        union_set = set(union)
        resident.extend(
            source
            for source in range(S - Q)
            if source not in union_set and len(resident) < C
        )
        if len(resident) != C or len(set(resident)) != C:
            raise RuntimeError("Cannot construct a full unique resident map")
        maps[request, torch.tensor(resident, dtype=torch.long)] = torch.arange(
            C, dtype=torch.int32
        )
        missing_by_request.append(missing)
        union_sizes.append(len(union))
    return maps, missing_by_request, union_sizes


def validate(args, reference, original_map, expected_missing):
    _sync()
    ids = args[13].cpu().view(B, Q, K).long()
    slots = args[14].cpu().view(B, Q, K).long()
    route_counts = args[15].cpu().view(B, Q).tolist()
    miss_src = args[16].cpu().long()
    miss_dst = args[17].cpu().long()
    counts = args[18].cpu().tolist()
    final_map = args[12].cpu().long()
    ref = reference.cpu().view(B, Q, K).long()
    details = []
    for row in range(B):
        ref_union = set(ref[row].reshape(-1).tolist())
        true_missing = {s for s in ref_union if int(original_map[row, s]) < 0}
        if true_missing != expected_missing[row]:
            raise AssertionError(
                f"request {row}: constructed miss union differs from target"
            )
        if not 0 <= counts[row] <= C:
            raise AssertionError(f"request {row}: invalid plan count {counts[row]}")
        srcs, dsts = miss_src[row, : counts[row]], miss_dst[row, : counts[row]]
        if (
            len(set(srcs.tolist())) != counts[row]
            or len(set(dsts.tolist())) != counts[row]
        ):
            raise AssertionError(f"request {row}: duplicate plan source or destination")
        if not torch.all((dsts >= 0) & (dsts < C)):
            raise AssertionError(f"request {row}: invalid plan destination")
        plan_set = set(srcs.tolist())
        if not true_missing <= plan_set:
            raise AssertionError(f"request {row}: native union omitted a true miss")
        refresh = plan_set - true_missing
        if refresh:
            raise AssertionError(
                f"request {row}: expected zero refresh entries, got {sorted(refresh)}"
            )
        if counts[row] != len(true_missing):
            raise AssertionError(
                f"request {row}: copy count differs from true union misses"
            )
        for route in range(Q):
            actual_set = set(ids[row, route].tolist())
            expected_route_misses = len(actual_set & true_missing)
            if route_counts[row][route] != expected_route_misses:
                raise AssertionError(
                    f"request {row}, route {route}: miss count mismatch"
                )
            if actual_set != set(ref[row, route].tolist()):
                raise AssertionError(
                    f"request {row}, route {route}: LI/LIM selection mismatch"
                )
            if any(source < 0 or source >= S - Q + route + 1 for source in actual_set):
                raise AssertionError(f"request {row}, route {route}: causal violation")
            if not torch.all((slots[row, route] >= 0) & (slots[row, route] < C)):
                raise AssertionError(
                    f"request {row}, route {route}: invalid selected slot"
                )
            if not torch.equal(final_map[row, ids[row, route]], slots[row, route]):
                raise AssertionError(f"request {row}, route {route}: slot map mismatch")
        for src, dst in zip(srcs.tolist(), dsts.tolist()):
            if int(final_map[row, src]) != dst:
                raise AssertionError(
                    f"request {row}: union destination does not match final map"
                )
        details.append(
            {
                "request": row,
                "selected_union": len(ref_union),
                "true_union_misses": len(true_missing),
                "refresh_entries": len(refresh),
                "plan_entries": counts[row],
                "per_route_misses": route_counts[row],
            }
        )
    return details


def time_op(fn, *, reset=None):
    if reset is not None:
        reset()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--misses", type=int, required=True, choices=[410, 737])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument(
        "--only-lim",
        action="store_true",
        help="time LIM only; reference LI still runs outside timing",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.rounds < 3:
        parser.error(
            "warmup and iterations must be positive; use at least three rounds"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    inputs, input_hashes = make_inputs(args.seed)
    reference = vendor_li(inputs)
    _sync()
    original_map, expected_missing, union_sizes = make_resident_map(
        reference, args.misses, args.seed
    )
    resident_device = original_map.to(inputs[12].device)

    def reset():
        inputs[12].copy_(resident_device)

    reset()
    native_lim(inputs)
    details = validate(inputs, reference, original_map, expected_missing)
    print(
        "FIXTURE_VALIDATED "
        + json.dumps(
            {
                "target_misses": args.misses,
                "hit_rate_percent": 100 * (1 - args.misses / (Q * K)),
                "input_sha256": input_hashes,
                "per_request": details,
            }
        ),
        flush=True,
    )
    for _ in range(args.warmup):
        if not args.only_lim:
            vendor_li(inputs)
        reset()
        native_lim(inputs)
    _sync()
    rounds = []
    for index in range(args.rounds):
        li, lim = [], []
        for _ in range(args.iterations):
            if not args.only_lim:
                li.append(time_op(lambda: vendor_li(inputs)))
            lim.append(time_op(lambda: native_lim(inputs), reset=reset))
        details = validate(inputs, reference, original_map, expected_missing)
        rounds.append(
            {
                "round": index + 1,
                "li_us": li,
                "lim_us": lim,
                "li_summary": _summary(li) if li else None,
                "lim_summary": _summary(lim),
            }
        )
    _sync()
    li_all = [x for row in rounds for x in row["li_us"]]
    lim_all = [x for row in rounds for x in row["lim_us"]]
    payload = {
        "description": "operator-only eager NPU-event timing; reset before LIM start event",
        "shape": {
            "sequence": S,
            "batch": B,
            "queries_per_request": Q,
            "top_k": K,
            "cache_capacity": C,
            "index_heads": HEADS,
            "head_dim": 128,
            "dtype": "bfloat16",
        },
        "hit_rate_definition": "1 - unique selected union misses / (Q * K)",
        "target_union_misses": args.misses,
        "actual_hit_rate_percent": 100 * (1 - args.misses / (Q * K)),
        "union_sizes": union_sizes,
        "per_request": details,
        "li_summary": _summary(li_all) if li_all else None,
        "lim_summary": _summary(lim_all),
        "mean_overhead_us": (
            statistics.fmean(lim_all) - statistics.fmean(li_all) if li_all else None
        ),
        "rounds": rounds,
        "seed": args.seed,
        "input_sha256": input_hashes,
        "warmup": args.warmup,
        "operator_library": _loaded_kernel_libraries(),
        "script_sha256": _sha(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", None),
        "sgl_kernel_npu_version": getattr(sgl_kernel_npu, "__version__", None),
        "device_index": torch.npu.current_device(),
        "device_name": torch.npu.get_device_name(torch.npu.current_device()),
        "environment": {
            k: os.environ[k]
            for k in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
            )
            if k in os.environ
        },
        "timestamp_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "target_misses": args.misses,
                "actual_hit_rate_percent": payload["actual_hit_rate_percent"],
                "li_summary": payload["li_summary"],
                "lim_summary": payload["lim_summary"],
                "mean_overhead_us": payload["mean_overhead_us"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
