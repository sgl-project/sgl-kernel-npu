#!/usr/bin/env python3
"""Validate and summarize a completed three-arm A5 matrix."""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
from pathlib import Path
from typing import Any

from result_contract import ARMS, read_result


def _read_f32(result_path: Path, entry: dict[str, Any]) -> array.array:
    path = result_path.parent / entry["path"]
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, math.prod(entry["shape"]))
    return values


def _parity(
    left: array.array,
    right: array.array,
    *,
    atol: float,
    rtol: float,
    max_relative_l2: float,
) -> dict[str, Any]:
    if len(left) != len(right):
        return {"passed": False, "reason": "shape/length mismatch"}
    square = 0.0
    reference_square = 0.0
    max_abs = 0.0
    close = True
    for actual, expected in zip(left, right):
        error = float(actual) - float(expected)
        max_abs = max(max_abs, abs(error))
        square += error * error
        reference_square += float(expected) * float(expected)
        close = close and abs(error) <= atol + rtol * abs(float(expected))
    relative_l2 = math.sqrt(square) / max(math.sqrt(reference_square), 1e-12)
    return {
        "passed": close and relative_l2 <= max_relative_l2,
        "atol": atol,
        "rtol": rtol,
        "max_relative_l2": max_relative_l2,
        "max_abs": max_abs,
        "rms": math.sqrt(square / max(len(left), 1)),
        "relative_l2": relative_l2,
    }


def build_report(
    root: Path, *, parity_atol: float, parity_rtol: float, max_relative_l2: float
) -> dict[str, Any]:
    indexed: dict[tuple[int, str], dict[str, tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(root.rglob("result.json")):
        try:
            result = read_result(path)
        except ValueError:
            continue
        case = (result["case"]["query_width"], result["case"]["mode"])
        indexed.setdefault(case, {})[result["arm"]["name"]] = (path, result)
    if not indexed:
        raise ValueError("no complete result.json files found")
    rows = []
    all_passed = True
    for case, arms in sorted(indexed.items()):
        if set(arms) != ARMS:
            raise ValueError(f"case {case} does not contain exactly three arms")
        fixtures = {
            (result["fixture"]["spec_sha256"], result["fixture"]["tensor_sha256"])
            for _, result in arms.values()
        }
        fixture_equal = len(fixtures) == 1
        native_path, native = arms["native"]
        for arm_name in ("native", "pre_migration", "post_migration"):
            path, result = arms[arm_name]
            parity = {}
            if arm_name != "native":
                for output_name in ("standalone", "chain"):
                    parity[output_name] = _parity(
                        _read_f32(path, result["outputs"][output_name]),
                        _read_f32(native_path, native["outputs"][output_name]),
                        atol=parity_atol,
                        rtol=parity_rtol,
                        max_relative_l2=max_relative_l2,
                    )
            row_passed = (
                result["status"] == "passed"
                and fixture_equal
                and all(value["passed"] for value in parity.values())
            )
            all_passed = all_passed and row_passed
            rows.append(
                {
                    "query_width": case[0],
                    "mode": case[1],
                    "arm": arm_name,
                    "fixture_equal": fixture_equal,
                    "status": result["status"],
                    "passed": row_passed,
                    "indexer_or_lim_median_us": result["timings"]["indexer_or_lim"][
                        "median_us"
                    ],
                    "attention_or_copysfa_median_us": result["timings"][
                        "attention_or_copysfa"
                    ]["median_us"],
                    "chain_median_us": result["timings"]["chain"]["median_us"],
                    "live_chain_lim": result["diagnostics"]["live_chain_lim"],
                    "parity_to_native": parity,
                    "result": str(path.relative_to(root)),
                }
            )
    return {
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parity-atol", type=float, default=0.002)
    parser.add_argument("--parity-rtol", type=float, default=0.005)
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    args = parser.parse_args()
    report = build_report(
        args.root,
        parity_atol=args.parity_atol,
        parity_rtol=args.parity_rtol,
        max_relative_l2=args.max_relative_l2,
    )
    (args.root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "query_width",
            "mode",
            "arm",
            "status",
            "passed",
            "indexer_or_lim_median_us",
            "attention_or_copysfa_median_us",
            "chain_median_us",
            "fallback_any",
            "miss_counts",
            "result",
        )
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in report["rows"]:
            flat = dict(row)
            live = row["live_chain_lim"] or {}
            flat["fallback_any"] = live.get("fallback_any")
            flat["miss_counts"] = json.dumps(live.get("miss_counts"))
            writer.writerow({field: flat[field] for field in fields})
    for row in report["rows"]:
        live = row["live_chain_lim"] or {}
        print(
            "Q{query_width} {mode:5} {arm:14} index/LIM={indexer_or_lim_median_us:.3f}us "
            "attention/CopySFA={attention_or_copysfa_median_us:.3f}us "
            "chain={chain_median_us:.3f}us fallback_any={fallback} status={status}".format(
                **row, fallback=live.get("fallback_any")
            )
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
