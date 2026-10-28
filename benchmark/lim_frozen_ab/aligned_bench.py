#!/usr/bin/env python3
"""Same-process A/B of original and ported LIM on unmodified frozen inputs."""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

B, Q, K, C, L, ACTUAL = 16, 4, 2048, 12288, 65536, 65664
INVALID = -(1 << 31)
FILES = {
    "shared.pt": "8aecdbfc648236ef851dbe317811028ca5c652ab98e4f426b11525623f22abd8",
    "cache-hit95.pt": "e8c5257b1c4889e84d4876766dc9a26cd3ced7fd171629bd047c24237a2e46d6",
    "cache-hit91.pt": "ebb129b3607d41d104a675ec29d75efe283b7ec903f14d442b2cb994b5e02c7f",
}
EXPECTED = {
    "query": "f359e8df224013944a99647f7eb655226e30dc1b9b96a97b4af472cce4e53686",
    "key": "4ef400f338b562afda0eed5acf55e7f1f65c130ebf0e84ac6ec3e79e105f727f",
    "weights": "e12a99c931afb4546eb538ffa35a2e7fcf8c3030e56a33c0f4b5a3e1c48e3e6b",
    "native_topk": "86f211145006539f3b96530bd03102fe82789f6d4d3b51d9afb3e0fef41f2f69",
}
NAMES = [
    "index_weights",
    "query_dequant_scale",
    "query",
    "index_key_dequant_scale",
    "index_key_cache",
    "index_block_table",
    "actual_seq_lengths_query",
    "actual_seq_lengths_key",
    "offload_seq_lengths_key",
    "num_cache_tokens",
    "request_state",
    "req_pool_entries",
    "cache_slots_pool",
    "topk_src_ids",
    "topk_dst_slots",
    "topk_miss_counts",
    "miss_src_ids",
    "miss_dst_slots",
    "miss_counts",
]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def stats(xs):
    return {
        "count": len(xs),
        "mean_us": statistics.fmean(xs),
        "median_us": statistics.median(xs),
        "p95_us": float(np.percentile(xs, 95)),
        "samples_us": xs,
    }


def libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if (
            path.startswith("/")
            and ".so" in path
            and any(x in path.lower() for x in ("sgl_kernel_npu", "ascend", "opapi"))
            and Path(path).is_file()
        ):
            paths.add(path)
    return [{"path": p, "sha256": file_hash(Path(p))} for p in sorted(paths)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--case", choices=["95", "91"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-dir", type=Path)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timing", choices=["eager", "graph", "both"], default="both")
    args = p.parse_args()
    if args.output.exists() or (args.profile_dir and args.profile_dir.exists()):
        raise FileExistsError("Output or profile directory already exists")
    if min(args.warmup, args.iterations, args.rounds) < 1:
        p.error("warmup, iterations and rounds must be positive")
    paths = [args.input_dir / "shared.pt", args.input_dir / f"cache-hit{args.case}.pt"]
    hashes = {x.name: file_hash(x) for x in paths}
    for name, value in hashes.items():
        if value != FILES[name]:
            raise ValueError(f"Frozen file hash mismatch: {name} {value}")
    shared, case = [torch.load(x, map_location="cpu", weights_only=True) for x in paths]
    cpu_hashes = {
        name: tensor_hash(tensor)
        for name, tensor in shared.items()
        if isinstance(tensor, torch.Tensor)
    }
    case_hashes = {
        name: tensor_hash(tensor)
        for name, tensor in case.items()
        if isinstance(tensor, torch.Tensor)
    }
    for name, value in EXPECTED.items():
        if cpu_hashes[name] != value:
            raise ValueError(f"Frozen tensor hash mismatch: {name}")
    if case_hashes["native_topk"] != EXPECTED["native_topk"]:
        raise ValueError("Frozen case TopK differs from shared")
    for name, shape, dtype in [
        ("query", (64, 32, 128), torch.bfloat16),
        ("weights", (64, 32), torch.bfloat16),
        ("key", (8208, 128, 1, 128), torch.bfloat16),
        ("native_topk", (64, 2048), torch.int32),
    ]:
        assert shared[name].shape == shape and shared[name].dtype == dtype, name
    seed_cpu = case["seed_pool"]
    assert seed_cpu.shape == (B, ACTUAL) and seed_cpu.dtype == torch.int32
    checks = {
        "index_block_table": torch.arange(B * 513, dtype=torch.int32).reshape(B, 513),
        "actual_seq_lengths_query": torch.arange(4, 65, 4, dtype=torch.int32),
        "actual_seq_lengths_key": torch.full((B,), ACTUAL, dtype=torch.int32),
        "offload_seq_lengths_key": torch.full((B,), L, dtype=torch.int32),
        "num_cache_tokens": torch.full((B,), C, dtype=torch.int32),
        "request_state": torch.full((B,), -1, dtype=torch.int32),
        "req_pool_entries": torch.arange(B, dtype=torch.int32),
    }
    for name, expected in checks.items():
        assert torch.equal(shared[name], expected), name
    assert torch.count_nonzero(shared["query_dequant_scale"]) == 0
    assert torch.count_nonzero(shared["index_key_dequant_scale"]) == 0
    reference = shared["native_topk"].numpy().reshape(B, Q, K)
    seed_np = seed_cpu.numpy()
    miss_sets, expected_route, expected_union = [], [], []
    for b in range(B):
        row = seed_np[b]
        resident = row[row != INVALID]
        assert len(resident) == C and np.array_equal(np.sort(resident), np.arange(C))
        assert np.all(row[L:] == INVALID)
        routes = reference[b]
        assert all(
            len(set(map(int, route))) == K and np.all((route >= 0) & (route < L))
            for route in routes
        )
        union = np.unique(routes)
        missing = set(map(int, union[row[union] == INVALID]))
        miss_sets.append(missing)
        expected_union.append(len(missing))
        expected_route.append(
            [len(missing.intersection(map(int, route))) for route in routes]
        )
    assert np.array_equal(expected_union, case["union_miss_counts"].numpy())
    assert np.array_equal(expected_route, case["query_miss_counts"].numpy())
    # Register vLLM custom OPP and both Torch namespaces before NPU initialization.
    if not enable_custom_op():
        raise RuntimeError("vLLM custom op unavailable")
    import sgl_kernel_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    d = {
        k: v.to(dev)
        for k, v in shared.items()
        if isinstance(v, torch.Tensor) and k != "native_topk"
    }
    seed = seed_cpu.to(dev)
    pool = torch.empty_like(seed)
    src = torch.empty((B * Q, 1, K), dtype=torch.int32, device=dev)
    dst = torch.empty_like(src)
    route_counts = torch.empty((B * Q,), dtype=torch.int32, device=dev)
    miss_src = torch.empty((B, 32768), dtype=torch.int32, device=dev)
    miss_dst = torch.empty_like(miss_src)
    counts = torch.empty((B,), dtype=torch.int32, device=dev)
    tensors = [
        d["weights"],
        d["query_dequant_scale"],
        d["query"],
        d["index_key_dequant_scale"],
        d["key"],
        d["index_block_table"],
        d["actual_seq_lengths_query"],
        d["actual_seq_lengths_key"],
        d["offload_seq_lengths_key"],
        d["num_cache_tokens"],
        d["request_state"],
        d["req_pool_entries"],
        pool,
        src,
        dst,
        route_counts,
        miss_src,
        miss_dst,
        counts,
    ]

    def original():
        torch.ops._C_ascend.npu_fused_lightning_indexer_manage(
            **dict(zip(NAMES, tensors))
        )

    def port():
        torch.ops.npu.fused_lightning_indexer_manage(*tensors, full_visible=False)

    def li():
        return torch_npu.npu_lightning_indexer(
            query=d["query"],
            key=d["key"],
            weights=d["weights"],
            actual_seq_lengths_query=d["actual_seq_lengths_query"],
            actual_seq_lengths_key=d["offload_seq_lengths_key"],
            block_table=d["index_block_table"],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=K,
            sparse_mode=0,
        )

    def topk(result):
        return result[0] if isinstance(result, (list, tuple)) else result

    def validate(full=True):
        torch.npu.synchronize()
        if counts.cpu().tolist() != expected_union:
            raise AssertionError("Per-request native union count differs from frozen")
        if not full:
            return
        s = src.cpu().numpy().reshape(B, Q, K)
        t = dst.cpu().numpy().reshape(B, Q, K)
        r = route_counts.cpu().numpy().reshape(B, Q)
        ms = miss_src.cpu().numpy()
        md = miss_dst.cpu().numpy()
        final = pool.cpu().numpy()
        for b in range(B):
            n = expected_union[b]
            sources = ms[b, :n]
            slots = md[b, :n]
            row = final[b]
            assert (
                set(map(int, sources)) == miss_sets[b]
                and len(set(map(int, sources))) == n
            )
            assert len(set(map(int, slots))) == n and np.all((slots >= 0) & (slots < C))
            assert np.array_equal(row[sources], slots)
            resident = row[row != INVALID]
            assert (
                len(resident) == C
                and np.array_equal(np.sort(resident), np.arange(C))
                and np.all(row[L:] == INVALID)
            )
            for q in range(Q):
                assert set(map(int, s[b, q])) == set(map(int, reference[b, q]))
                assert np.array_equal(row[s[b, q]], t[b, q])
                assert r[b, q] == expected_route[b][q]
                assert set(map(int, s[b, q, : r[b, q]])) == miss_sets[b].intersection(
                    map(int, reference[b, q])
                )

    actual_li = topk(li()).cpu().numpy().reshape(B, Q, K)
    for b in range(B):
        for q in range(Q):
            assert set(map(int, actual_li[b, q])) == set(map(int, reference[b, q]))
    for fn in (original, port):
        pool.copy_(seed)
        fn()
        validate()
    loaded = libraries()
    sgl_lib = [
        item for item in loaded if Path(item["path"]).name == "libsgl_kernel_npu.so"
    ]
    if (
        len(sgl_lib) != 1
        or sgl_lib[0]["sha256"]
        != "5ff20ae94d92779c87a9425097c88621063b7dc52387be7a61a449a77da0ab80"
    ):
        raise RuntimeError(f"Unexpected SGLang kernel library: {sgl_lib}")
    addresses = {name: int(t.data_ptr()) for name, t in zip(NAMES, tensors)}
    addresses["seed_pool"] = int(seed.data_ptr())
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
        enable_timing=True
    )

    def timed(fn, reset=False):
        if reset:
            pool.copy_(seed)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end) * 1000)

    runners = {"eager": {"li": li, "original": original, "sglang": port}}
    graphs = []
    if args.timing in ("graph", "both"):
        current = torch.npu.current_stream()
        capture = torch.npu.Stream()
        capture.wait_stream(current)
        with torch.npu.stream(capture):
            for fn in (li, original, port):
                pool.copy_(seed)
                fn()
            torch.npu.synchronize()
            gs = {}
            for name, fn in (("li", li), ("original", original), ("sglang", port)):
                pool.copy_(seed)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=capture):
                    value = fn()
                gs[name] = graph
                if name == "li":
                    graph_li = value
                graphs.append(graph)
        current.wait_stream(capture)
        gs["li"].replay()
        gl = topk(graph_li).cpu().numpy().reshape(B, Q, K)
        for b in range(B):
            for q in range(Q):
                assert set(map(int, gl[b, q])) == set(map(int, reference[b, q]))
        for name in ("original", "sglang"):
            pool.copy_(seed)
            gs[name].replay()
            validate()
        runners["graph"] = {name: g.replay for name, g in gs.items()}
    payload = {
        "case": args.case,
        "file_sha256": hashes,
        "input_tensor_sha256": cpu_hashes,
        "case_tensor_sha256": case_hashes,
        "shape": {
            "B": B,
            "Q": Q,
            "K": K,
            "C": C,
            "actual_kv": ACTUAL,
            "prefix": L,
            "state": -1,
            "li_sparse_mode": 0,
        },
        "expected_union_misses": expected_union,
        "expected_query_misses": expected_route,
        "actual_hit_rate": 1 - sum(map(sum, expected_route)) / (B * Q * K),
        "tensor_addresses": addresses,
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device_name": torch.npu.get_device_name(torch.npu.current_device()),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "ASCEND_CUSTOM_OPP_PATH",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
            )
        },
        "script_sha256": file_hash(Path(__file__)),
        "loaded_shared_objects": libraries(),
        "results": {},
        "errors": {},
    }
    modes = ("eager", "graph") if args.timing == "both" else (args.timing,)
    for mode in modes:
        fns = runners[mode]
        rounds = []
        for rnd in range(args.rounds):
            for i in range(args.warmup):
                for name in (
                    ("original", "sglang")
                    if (i + rnd) % 2 == 0
                    else ("sglang", "original")
                ):
                    timed(fns[name], True)
                    validate(False)
                timed(fns["li"])
            samples = {"li": [], "original": [], "sglang": []}
            orders = []
            for i in range(args.iterations):
                samples["li"].append(timed(fns["li"]))
                order = (
                    ("original", "sglang")
                    if (i + rnd) % 2 == 0
                    else ("sglang", "original")
                )
                orders.append(list(order))
                for name in order:
                    samples[name].append(timed(fns[name], True))
                    validate(i == args.iterations - 1)
            delta = [s - v for s, v in zip(samples["sglang"], samples["original"])]
            rounds.append(
                {
                    "round": rnd + 1,
                    "order": orders,
                    "li": stats(samples["li"]),
                    "original": stats(samples["original"]),
                    "sglang": stats(samples["sglang"]),
                    "paired_sglang_minus_original": stats(delta),
                }
            )
        payload["results"][mode] = rounds
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True)
        for mode in modes:
            fns = runners[mode]
            dest = args.profile_dir / mode
            dest.mkdir()
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(dest)),
            ) as prof:
                for _ in range(3):
                    with torch.autograd.profiler.record_function("LI"):
                        fns["li"]()
                    for name in ("original", "sglang"):
                        with torch.autograd.profiler.record_function("reset_" + name):
                            pool.copy_(seed)
                        with torch.autograd.profiler.record_function(name):
                            fns[name]()
                    prof.step()
                torch.npu.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("ALIGNED_SUCCESS", args.output, flush=True)


if __name__ == "__main__":
    main()
