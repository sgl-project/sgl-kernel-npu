"""Versioned result checks and cross-process comparison helpers."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ARMS = {"native", "pre_migration", "post_migration"}
KINDS = {"native", "lim_copysfa"}
MODES = {"eager", "graph"}
COMPONENTS = {"indexer_or_lim", "attention_or_copysfa", "chain"}


def timing_stats(samples_us: list[float]) -> dict[str, Any]:
    if not samples_us or any(
        not math.isfinite(value) or value < 0 for value in samples_us
    ):
        raise ValueError("timing samples must be finite non-negative values")
    ordered = sorted(samples_us)
    return {
        "samples_us": samples_us,
        "count": len(samples_us),
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.fmean(samples_us),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "p95_us": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
    }


def validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported result schema_version")
    if result.get("arm", {}).get("name") not in ARMS:
        raise ValueError("unknown arm name")
    if result.get("arm", {}).get("kind") not in KINDS:
        raise ValueError("unknown arm kind")
    if result.get("case", {}).get("mode") not in MODES:
        raise ValueError("mode must be eager or graph")
    if result.get("timer") != "npu_event":
        raise ValueError("A5 performance results must use NPU event timing")
    profiling = result.get("profiling")
    if not isinstance(profiling, dict) or not isinstance(
        profiling.get("enabled"), bool
    ):
        raise ValueError("result must record optional profiling state")
    if profiling["enabled"]:
        if set(profiling.get("components", {})) != COMPONENTS:
            raise ValueError("enabled profiling must contain all three components")
        for name, value in profiling["components"].items():
            for field in ("trace_dir", "iterations", "reset_marker", "op_marker"):
                if not value.get(field):
                    raise ValueError(
                        f"missing profiling component field {name}.{field}"
                    )
    elif profiling.get("components"):
        raise ValueError("disabled profiling must not claim component traces")
    timings = result.get("timings", {})
    if set(timings) != COMPONENTS:
        raise ValueError("result must contain indexer/LIM, attention/CopySFA and chain")
    for name, value in timings.items():
        samples = value.get("samples_us")
        if not isinstance(samples, list) or value.get("count") != len(samples):
            raise ValueError(f"invalid samples for {name}")
        expected = timing_stats(samples)
        for field in ("median_us", "mean_us", "min_us", "max_us", "p95_us"):
            if not math.isclose(
                value[field], expected[field], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"incorrect {field} for {name}")
    fixture = result.get("fixture", {})
    for field in ("spec_sha256", "tensor_sha256", "miss_counts"):
        if not fixture.get(field):
            raise ValueError(f"missing fixture.{field}")
    provenance = result.get("provenance", {})
    if not provenance.get("framework_prefix") or not provenance.get("kernel_prefix"):
        raise ValueError("framework/kernel prefixes are required provenance")
    if "schemas" not in provenance or "libraries" not in provenance:
        raise ValueError("operator schemas and library hashes are required")
    accuracy = result.get("accuracy", {})
    for field in ("standalone", "chain", "canonical_ids"):
        if field not in accuracy:
            raise ValueError(f"missing accuracy.{field}")
    diagnostics = result.get("diagnostics", {})
    fixed = diagnostics.get("fixed_attention_plan")
    if not fixed or "topk_misses" not in fixed or "fallback_any" not in fixed:
        raise ValueError("fixed standalone attention-plan diagnostics are required")
    live = diagnostics.get("live_chain_lim")
    if result["arm"]["kind"] == "lim_copysfa":
        if not live or any(
            field not in live
            for field in (
                "topk_misses",
                "miss_counts",
                "fallback_any",
                "matches_fixture_union_misses",
                "selected_prefix_hit_rates",
            )
        ):
            raise ValueError("LIM/CopySFA arms require live chain miss diagnostics")
    elif live is not None:
        raise ValueError("native arm must not claim LIM miss diagnostics")
    if result.get("status") not in {"passed", "failed"}:
        raise ValueError("status must be passed or failed")


def read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_result(value)
    return value


def write_result(path: Path, result: dict[str, Any]) -> None:
    validate_result(result)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
