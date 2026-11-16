#!/usr/bin/env python3
"""Run one isolated A5 benchmark arm with a frozen cross-arm fixture."""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from fixture import FixtureSpec, build_torch_fixture, parse_miss_counts, unpack_rows
from result_contract import SCHEMA_VERSION, timing_stats, write_result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _shared_object_hashes(root: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []
    base = root if root.is_dir() else root.parent
    candidates = [root] if root.is_file() else sorted(root.rglob("*.so"))
    return [
        {
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(base)),
            "sha256": _sha256(path),
        }
        for path in candidates
        if path.is_file()
    ]


def _add_prefix(path: Path) -> None:
    path = path.resolve(strict=True)
    for candidate in (path / "python", path):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _schemas(torch: Any, fragments: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in torch._C._dispatch_get_all_op_names():
        if any(fragment in name for fragment in fragments):
            try:
                result[name] = str(
                    torch._C._dispatch_find_schema_or_throw(name, "").schema()
                )
            except RuntimeError:
                result[name] = "schema lookup failed"
    return result


def _first_tensor(value: Any) -> Any:
    return value[0] if isinstance(value, tuple) else value


def _require_kernel_prefix(module_file: str, prefix: Path) -> None:
    if not Path(module_file).resolve().is_relative_to(prefix.resolve()):
        raise RuntimeError(
            f"sgl_kernel_npu resolved outside --kernel-prefix: {module_file}; expected {prefix}"
        )


def _metric(
    torch: Any,
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
    max_relative_l2: float,
) -> dict[str, Any]:
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    error = actual - expected
    finite = bool(torch.isfinite(actual).all())
    max_abs = float(error.abs().max())
    rms = float(error.square().mean().sqrt())
    relative_l2 = float(error.norm() / expected.norm().clamp_min(1e-12))
    return {
        "finite": finite,
        "max_abs": max_abs,
        "rms": rms,
        "relative_l2": relative_l2,
        "atol": atol,
        "rtol": rtol,
        "max_relative_l2": max_relative_l2,
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "passed": finite
        and bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
        and relative_l2 <= max_relative_l2,
    }


def _canonical_metric(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    actual = actual.detach().cpu().to(torch.int64).reshape(expected.shape)
    expected = expected.detach().cpu().to(torch.int64)
    ordered_actual = actual.sort(dim=-1).values
    ordered_expected = expected.sort(dim=-1).values
    mismatch = ordered_actual != ordered_expected
    return {
        "passed": not bool(mismatch.any()),
        "mismatch_values": int(mismatch.sum()),
        "rows": expected.shape[0],
        "sha256": hashlib.sha256(
            ordered_actual.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def _write_raw_tensor(torch: Any, path: Path, tensor: Any) -> dict[str, Any]:
    cpu = tensor.detach().cpu().float().contiguous()
    values = array.array("f", cpu.reshape(-1).tolist())
    with path.open("wb") as stream:
        values.tofile(stream)
    return {"path": path.name, "shape": list(cpu.shape), "sha256": _sha256(path)}


def _make_graph_callable(
    torch: Any, enqueue: Callable[[], None], reset: Callable[[], None]
):
    reset()
    enqueue()
    torch.npu.synchronize()
    reset()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        enqueue()
    torch.npu.synchronize()
    return graph.replay, graph


def _prepare_callable(
    torch: Any,
    enqueue: Callable[[], None],
    reset: Callable[[], None],
    mode: str,
):
    if mode == "graph":
        return _make_graph_callable(torch, enqueue, reset)
    reset()
    enqueue()
    torch.npu.synchronize()
    return enqueue, None


def _time_component(
    torch: Any,
    operation: Callable[[], None],
    reset: Callable[[], None],
    warmup: int,
    rounds: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        reset()
        operation()
    torch.npu.synchronize()
    samples: list[float] = []
    round_medians: list[float] = []
    for _ in range(rounds):
        round_samples: list[float] = []
        for _ in range(iterations):
            reset()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            value = float(start.elapsed_time(end)) * 1000.0
            round_samples.append(value)
            samples.append(value)
        round_medians.append(float(timing_stats(round_samples)["median_us"]))
    return timing_stats(samples) | {"round_medians_us": round_medians}


def _profile_component(
    torch: Any,
    torch_npu: Any,
    *,
    component: str,
    operation: Callable[[], None],
    reset: Callable[[], None],
    iterations: int,
    trace_dir: Path,
) -> dict[str, Any]:
    """Collect an independent post-timing NPU trace for one component."""
    from torch_npu.profiler import ProfilerActivity, profile

    trace_dir.mkdir(parents=True, exist_ok=False)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    reset_marker = f"a5_three_arm::{component}::reset"
    op_marker = f"a5_three_arm::{component}::op"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(trace_dir), async_mode=True
        ),
        experimental_config=experimental_config,
    ) as profiler:
        for _ in range(iterations):
            with torch.profiler.record_function(reset_marker):
                reset()
                torch.npu.synchronize()
            with torch.profiler.record_function(op_marker):
                operation()
                torch.npu.synchronize()
            profiler.step()
    files = sorted(
        str(path.relative_to(trace_dir))
        for path in trace_dir.rglob("*")
        if path.is_file()
    )
    return {
        "trace_dir": str(trace_dir.resolve()),
        "iterations": iterations,
        "reset_marker": reset_marker,
        "op_marker": op_marker,
        "files_observed_at_exit": files,
    }


def _dynamic_golden(torch: Any, fixture: dict[str, Any], ids: Any) -> Any:
    tensors = fixture["tensors"]
    packed = tensors["packed"].view(-1, 656)
    queries = tensors["merged_query"]
    ids_cpu = ids.detach().cpu().long().reshape(queries.shape[0], -1)
    output = torch.empty(queries.shape[0], queries.shape[1], 512, dtype=torch.float32)
    for row in range(queries.shape[0]):
        key, value = unpack_rows(torch, packed[ids_cpu[row]])
        scores = queries[row].float() @ key.T / math.sqrt(576)
        output[row] = torch.softmax(scores, dim=-1) @ value
    return output


def _import_shm():
    from sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm import (
        create_fused_shm_tensor,
        free_fused_shm,
    )

    return create_fused_shm_tensor, free_fused_shm


def _build_native(torch: Any, torch_npu: Any, fixture: dict[str, Any], device: Any):
    cpu = fixture["tensors"]
    values = {
        name: value.to(device) for name, value in cpu.items() if hasattr(value, "to")
    }
    holder: dict[str, Any] = {}

    def indexer():
        holder["ids"] = _first_tensor(
            torch_npu.npu_quant_lightning_indexer(
                query=values["index_query"],
                key=values["index_key"],
                weights=values["index_weights"],
                query_dequant_scale=values["query_scale"],
                key_dequant_scale=values["key_scale"],
                query_quant_mode=0,
                key_quant_mode=0,
                actual_seq_lengths_query=values["query_ends"],
                actual_seq_lengths_key=values["seq_lens"],
                block_table=values["source_table"],
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        )

    def attention(ids: Any):
        result = torch_npu.npu_kv_quant_sparse_flash_attention(
            query=values["merged_query"],
            key=values["packed"].view(torch.float8_e4m3fn),
            value=values["packed"].view(torch.float8_e4m3fn),
            sparse_indices=ids,
            scale_value=1 / math.sqrt(576),
            key_quant_mode=2,
            value_quant_mode=2,
            key_dequant_scale=None,
            value_dequant_scale=None,
            actual_seq_lengths_query=values["query_ends"],
            actual_seq_lengths_kv=values["seq_lens"],
            block_table=values["source_table"],
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
        )
        return _first_tensor(result)

    def standalone_attention():
        holder["standalone_output"] = attention(values["canonical_ids"])

    def chain():
        indexer()
        holder["chain_output"] = attention(holder["ids"])

    def no_reset():
        pass

    return {
        "indexer": indexer,
        "attention": standalone_attention,
        "chain": chain,
        "reset_indexer": no_reset,
        "reset_attention": no_reset,
        "reset_chain": no_reset,
        "holder": holder,
        "values": values,
        "live_topk_misses": None,
        "live_miss_counts": None,
        "cleanup": no_reset,
    }


def _build_lim_copysfa(
    torch: Any, fixture: dict[str, Any], device: Any, device_id: int
):
    cpu = fixture["tensors"]
    values = {
        name: value.to(device) for name, value in cpu.items() if hasattr(value, "to")
    }
    create_shm, free_shm = _import_shm()
    owner, _, pointer = create_shm(cpu["packed"].shape, torch.int8, device_id=device_id)
    owner.copy_(cpu["packed"])
    lim = torch.ops.npu.fused_quant_lightning_indexer_manage.default
    copy = torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention.default
    batch = fixture["spec"].batch
    rows = batch * fixture["spec"].query_width
    prefix = fixture["spec"].prefix
    capacity = fixture["spec"].capacity
    holder: dict[str, Any] = {}
    sources = torch.full((rows, 1, 2048), -1, dtype=torch.int32, device=device)
    destinations = torch.full_like(sources, -1)
    route_misses = torch.zeros(rows, dtype=torch.int32, device=device)
    miss_sources = torch.full((batch, 1, 32768), -1, dtype=torch.int32, device=device)
    miss_slots = torch.full_like(miss_sources, -1)
    miss_counts = torch.zeros(batch, dtype=torch.int32, device=device)
    slot_pool = values["slot_pool_initial"].clone()
    request_state = torch.full((batch,), -1, dtype=torch.int32, device=device)
    prefix_lens = torch.full((batch,), prefix, dtype=torch.int32, device=device)
    cache_sizes = torch.full((batch,), capacity, dtype=torch.int32, device=device)
    hbm = values["initial_hbm"].clone()
    standalone_output = torch.empty(
        rows,
        fixture["spec"].heads,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    chain_output = torch.empty_like(standalone_output)

    def reset_lim():
        slot_pool.copy_(values["slot_pool_initial"])
        sources.fill_(-1)
        destinations.fill_(-1)
        route_misses.zero_()
        miss_sources.fill_(-1)
        miss_slots.fill_(-1)
        miss_counts.zero_()

    def indexer():
        lim(
            values["index_weights"],
            values["query_scale"],
            values["index_query"],
            values["key_scale"],
            values["index_key"],
            values["source_table"],
            values["query_ends"],
            values["seq_lens"],
            prefix_lens,
            cache_sizes,
            request_state,
            values["request_slots"],
            slot_pool,
            sources,
            destinations,
            route_misses,
            miss_sources,
            miss_slots,
            miss_counts,
        )
        holder["ids"] = sources

    def reset_attention():
        hbm.copy_(values["initial_hbm"])
        standalone_output.fill_(float("nan"))

    def standalone_attention_enqueue():
        copy(
            values["merged_query"],
            hbm,
            owner,
            pointer,
            values["hbm_table"],
            values["source_table"],
            values["destination_slots"],
            values["source_ids"],
            values["route_misses"],
            values["query_ends"],
            values["seq_lens"],
            values["hbm_lengths"],
            values["miss_sources"],
            values["miss_slots"],
            values["miss_counts"],
            1 / math.sqrt(576),
            standalone_output,
        )
        holder["standalone_output"] = standalone_output

    def reset_chain():
        reset_lim()
        hbm.copy_(values["initial_hbm"])
        chain_output.fill_(float("nan"))

    def chain():
        indexer()
        copy(
            values["merged_query"],
            hbm,
            owner,
            pointer,
            values["hbm_table"],
            values["source_table"],
            destinations,
            sources,
            route_misses,
            values["query_ends"],
            values["seq_lens"],
            values["hbm_lengths"],
            miss_sources,
            miss_slots,
            miss_counts,
            1 / math.sqrt(576),
            chain_output,
        )
        holder["chain_output"] = chain_output

    def cleanup():
        torch.npu.synchronize()
        free_shm()

    return {
        "indexer": indexer,
        "attention": standalone_attention_enqueue,
        "chain": chain,
        "reset_indexer": reset_lim,
        "reset_attention": reset_attention,
        "reset_chain": reset_chain,
        "holder": holder,
        "values": values,
        "live_topk_misses": route_misses,
        "live_miss_counts": miss_counts,
        "cleanup": cleanup,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("native", "pre_migration", "post_migration"), required=True
    )
    parser.add_argument("--kind", choices=("native", "lim_copysfa"), required=True)
    parser.add_argument("--framework-prefix", type=Path, required=True)
    parser.add_argument("--kernel-prefix", type=Path, required=True)
    parser.add_argument("--library", type=Path, action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-width", type=int, choices=(1, 4), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--index-heads", type=int, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--miss-counts", default="512,513")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--golden-atol", type=float, default=0.005)
    parser.add_argument("--golden-rtol", type=float, default=0.02)
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.kind == "native" and args.arm != "native":
        raise SystemExit("only the native arm may use --kind native")
    if args.kind == "lim_copysfa" and args.arm == "native":
        raise SystemExit("the native arm must use --kind native")
    if min(args.warmup, args.rounds, args.iterations, args.profile_iterations) < 1:
        raise SystemExit(
            "warmup, rounds, iterations and profile-iterations must be positive"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    spec = FixtureSpec(
        query_width=args.query_width,
        batch=args.batch,
        heads=args.heads,
        index_heads=args.index_heads,
        capacity=args.capacity,
        prefix=args.prefix,
        context=args.context,
        miss_counts=parse_miss_counts(args.miss_counts),
    )
    spec.validate()
    _add_prefix(args.framework_prefix)
    _add_prefix(args.kernel_prefix)
    result_path = args.output / "result.json"
    partial = {
        "schema_version": SCHEMA_VERSION,
        "arm": {"name": args.arm, "kind": args.kind},
        "case": {"query_width": args.query_width, "mode": args.mode},
        "status": "running",
        "command": sys.argv,
    }
    result_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
    runtime = None
    try:
        # The package stages its pinned LIM OPP before importing torch_npu.
        # Importing torch_npu or initializing a device first can cache the
        # wrong custom-OPP discovery path in a fresh arm process.
        import sgl_kernel_npu

        _require_kernel_prefix(sgl_kernel_npu.__file__, args.kernel_prefix)
        import acl
        import torch
        import torch_npu

        torch.npu.set_device(args.device)
        if acl.rt.get_device_info(args.device, 601) != (3510, 0):
            raise RuntimeError("this benchmark requires Ascend950")
        libraries = []
        for library in args.library:
            resolved = library.resolve(strict=True)
            torch.ops.load_library(str(resolved))
            libraries.append({"path": str(resolved), "sha256": _sha256(resolved)})
        fixture = build_torch_fixture(torch, spec)
        device = torch.device("npu", args.device)
        runtime = (
            _build_native(torch, torch_npu, fixture, device)
            if args.kind == "native"
            else _build_lim_copysfa(torch, fixture, device, args.device)
        )

        prepared = {}
        graph_owners = []
        for name, reset_name in (
            ("indexer_or_lim", "reset_indexer"),
            ("attention_or_copysfa", "reset_attention"),
            ("chain", "reset_chain"),
        ):
            operation_name = {
                "indexer_or_lim": "indexer",
                "attention_or_copysfa": "attention",
                "chain": "chain",
            }[name]
            operation, graph = _prepare_callable(
                torch,
                runtime[operation_name],
                runtime[reset_name],
                args.mode,
            )
            prepared[name] = (operation, runtime[reset_name])
            if graph is not None:
                graph_owners.append(graph)

        timings = {
            name: _time_component(
                torch,
                operation,
                reset,
                args.warmup,
                args.rounds,
                args.iterations,
            )
            for name, (operation, reset) in prepared.items()
        }

        profiling = {
            "enabled": args.profile,
            "iterations_per_component": args.profile_iterations,
            "scope": "independent post-timing torch_npu profiler collection",
            "components": {},
        }
        if args.profile:
            profile_root = args.output / "profiles"
            for name, (operation, reset) in prepared.items():
                profiling["components"][name] = _profile_component(
                    torch,
                    torch_npu,
                    component=name,
                    operation=operation,
                    reset=reset,
                    iterations=args.profile_iterations,
                    trace_dir=profile_root / name,
                )

        for name, (operation, reset) in prepared.items():
            reset()
            operation()
            torch.npu.synchronize()
        holder = runtime["holder"]
        standalone = holder["standalone_output"]
        chain_output = holder["chain_output"]
        actual_ids = holder["ids"]
        metric_args = {
            "atol": args.golden_atol,
            "rtol": args.golden_rtol,
            "max_relative_l2": args.max_relative_l2,
        }
        standalone_metric = _metric(
            torch, standalone, fixture["tensors"]["golden"], **metric_args
        )
        chain_golden = _dynamic_golden(torch, fixture, actual_ids)
        chain_metric = _metric(torch, chain_output, chain_golden, **metric_args)
        canonical_metric = _canonical_metric(
            torch, actual_ids, fixture["tensors"]["canonical_ids"]
        )
        fixed_miss_counts = fixture["tensors"]["miss_counts"].tolist()
        fixed_topk_misses = fixture["tensors"]["route_misses"].tolist()
        fixed_hit_rates = fixture["manifest"]["selected_prefix_hit_rates"]
        if runtime["live_miss_counts"] is None:
            live_chain = None
        else:
            live_miss_counts = runtime["live_miss_counts"].detach().cpu().tolist()
            live_topk_misses = runtime["live_topk_misses"].detach().cpu().tolist()
            actual_ids_cpu = (
                actual_ids.detach()
                .cpu()
                .reshape(spec.batch, spec.query_width, spec.topk)
            )
            selected_prefix_unions = [
                len(
                    {
                        int(source)
                        for source in actual_ids_cpu[request].reshape(-1).tolist()
                        if 0 <= int(source) < spec.prefix
                    }
                )
                for request in range(spec.batch)
            ]
            selected_prefix_hit_rates = [
                1 - miss_count / union_count if union_count else 1.0
                for miss_count, union_count in zip(
                    live_miss_counts, selected_prefix_unions
                )
            ]
            live_chain = {
                "miss_counts": live_miss_counts,
                "topk_misses": live_topk_misses,
                "selected_prefix_union_counts": selected_prefix_unions,
                "selected_prefix_hit_rates": selected_prefix_hit_rates,
                "fallback_any": any(value > 512 for value in live_miss_counts),
                "fallback_requests": [
                    request
                    for request, value in enumerate(live_miss_counts)
                    if value > 512
                ],
                "matches_fixture_union_misses": live_miss_counts == fixed_miss_counts,
            }
        live_miss_passed = (
            live_chain is None or live_chain["matches_fixture_union_misses"]
        )
        outputs = {
            "standalone": _write_raw_tensor(
                torch, args.output / "standalone-output.f32", standalone
            ),
            "chain": _write_raw_tensor(
                torch, args.output / "chain-output.f32", chain_output
            ),
        }
        schemas = _schemas(
            torch,
            (
                "quant_lightning_indexer",
                "kv_quant_sparse_flash_attention",
                "fused_quant_lightning_indexer_manage",
                "fused_quant_scatter_copy_sparse_flash_attention",
            ),
        )
        package_root = (
            Path(sgl_kernel_npu.__file__).resolve().parent
            if sgl_kernel_npu is not None
            else None
        )
        try:
            package_version = importlib.metadata.version("sgl-kernel-npu")
        except importlib.metadata.PackageNotFoundError:
            package_version = None
        provenance = {
            "framework_prefix": str(args.framework_prefix.resolve()),
            "kernel_prefix": str(args.kernel_prefix.resolve()),
            "framework_sha": _git_sha(args.framework_prefix),
            "kernel_sha": _git_sha(args.kernel_prefix),
            "libraries": libraries,
            "schemas": schemas,
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
            "sgl_kernel_npu_file": (
                str(Path(sgl_kernel_npu.__file__).resolve())
                if sgl_kernel_npu is not None
                else None
            ),
            "sgl_kernel_npu_version": package_version,
            "sgl_kernel_npu_shared_objects": (
                _shared_object_hashes(package_root) if package_root else []
            ),
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "arm": {"name": args.arm, "kind": args.kind},
            "case": {
                "query_width": args.query_width,
                "mode": args.mode,
                "batch": args.batch,
                "heads": args.heads,
                "index_heads": args.index_heads,
                "capacity": args.capacity,
                "prefix": args.prefix,
                "context": args.context,
                "device": args.device,
            },
            "timer": "npu_event",
            "timing_boundary": {
                "indexer_or_lim": "one native indexer or LIM launch only",
                "attention_or_copysfa": "one native C8 attention or frozen-plan CopySFA launch only",
                "chain": "indexer+native attention, or LIM+CopySFA; cold reset excluded",
                "excluded": "fixture construction, host registration, allocation, cold reset, validation, output transfer",
            },
            "warmup": args.warmup,
            "rounds": args.rounds,
            "iterations": args.iterations,
            "fixture": {
                "spec_sha256": fixture["spec_sha256"],
                "tensor_sha256": fixture["tensor_sha256"],
                "miss_counts": fixture["manifest"]["miss_counts"],
                "manifest": fixture["manifest"],
            },
            "provenance": provenance,
            "timings": timings,
            "profiling": profiling,
            "accuracy": {
                "standalone": standalone_metric,
                "chain": chain_metric,
                "canonical_ids": canonical_metric,
            },
            "diagnostics": {
                "fixed_attention_plan": {
                    "scope": "standalone attention/CopySFA only; not LIM output",
                    "miss_counts": fixed_miss_counts,
                    "topk_misses": fixed_topk_misses,
                    "selected_prefix_union_count": fixture["manifest"][
                        "selected_prefix_union"
                    ],
                    "selected_prefix_hit_rates": fixed_hit_rates,
                    "fallback_any": any(value > 512 for value in fixed_miss_counts),
                },
                "live_chain_lim": live_chain,
            },
            "outputs": outputs,
            "status": (
                "passed"
                if standalone_metric["passed"]
                and chain_metric["passed"]
                and canonical_metric["passed"]
                and live_miss_passed
                else "failed"
            ),
        }
        write_result(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    except Exception:
        partial["status"] = "failed"
        partial["error"] = traceback.format_exc()
        result_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        traceback.print_exc()
        return 1
    finally:
        if runtime is not None:
            runtime["cleanup"]()


if __name__ == "__main__":
    raise SystemExit(main())
