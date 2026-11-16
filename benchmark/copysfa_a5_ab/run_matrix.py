#!/usr/bin/env python3
"""Launch native/pre/post A5 arms in isolated Python processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fixture import FixtureSpec, parse_miss_counts


def load_config(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("arms"), list):
        raise ValueError("arm config must use schema_version=1 and contain arms")
    arms = value["arms"]
    expected = {
        "native": "native",
        "pre_migration": "lim_copysfa",
        "post_migration": "lim_copysfa",
    }
    if {arm.get("name"): arm.get("kind") for arm in arms} != expected:
        raise ValueError(
            "config must contain native, pre_migration and post_migration exactly once"
        )
    for arm in arms:
        for field in ("python", "framework_prefix", "kernel_prefix"):
            if not arm.get(field):
                raise ValueError(f"{arm['name']} is missing {field}")
        if not isinstance(arm.get("libraries", []), list):
            raise ValueError(f"{arm['name']}.libraries must be a list")
    return arms


def build_command(
    args: argparse.Namespace, arm: dict[str, Any], width: int, mode: str, output: Path
) -> list[str]:
    command = [
        arm["python"],
        str(Path(__file__).with_name("bench.py")),
        "--arm",
        arm["name"],
        "--kind",
        arm["kind"],
        "--framework-prefix",
        arm["framework_prefix"],
        "--kernel-prefix",
        arm["kernel_prefix"],
        "--device",
        str(args.device),
        "--query-width",
        str(width),
        "--mode",
        mode,
        "--batch",
        str(args.batch),
        "--heads",
        str(args.heads),
        "--index-heads",
        str(args.index_heads),
        "--capacity",
        str(args.capacity),
        "--prefix",
        str(args.prefix),
        "--context",
        str(args.context),
        "--miss-counts",
        args.miss_counts,
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--iterations",
        str(args.iterations),
        "--golden-atol",
        str(args.golden_atol),
        "--golden-rtol",
        str(args.golden_rtol),
        "--max-relative-l2",
        str(args.max_relative_l2),
        "--output",
        str(output),
    ]
    for library in arm.get("libraries", []):
        command.extend(("--library", library))
    if args.profile:
        command.extend(
            ("--profile", "--profile-iterations", str(args.profile_iterations))
        )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--index-heads", type=int, required=True)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--prefix", type=int, default=9216)
    parser.add_argument("--context", type=int, default=9345)
    parser.add_argument("--miss-counts", default="512,513")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--golden-atol", type=float, default=0.005)
    parser.add_argument("--golden-rtol", type=float, default=0.02)
    parser.add_argument("--parity-atol", type=float, default=0.002)
    parser.add_argument("--parity-rtol", type=float, default=0.005)
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    arms = load_config(args.arm_config)
    for width in (1, 4):
        FixtureSpec(
            width,
            args.batch,
            args.heads,
            args.index_heads,
            args.capacity,
            args.prefix,
            args.context,
            miss_counts=parse_miss_counts(args.miss_counts),
        ).validate()
    if args.output_root.exists() and not args.dry_run:
        raise SystemExit(f"refusing to overwrite output root: {args.output_root}")
    commands = []
    for width in (1, 4):
        for mode in ("eager", "graph"):
            for arm in arms:
                output = args.output_root / f"q{width}-{mode}" / arm["name"]
                commands.append(build_command(args, arm, width, mode, output))
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return 0
    args.output_root.mkdir(parents=True)
    environment = os.environ.copy()
    # Device ownership is selected only by the explicit --device argument.
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
        environment.pop(name, None)
    receipts = []
    exit_code = 0
    for command in commands:
        completed = subprocess.run(command, env=environment)
        receipts.append({"command": command, "exit_code": completed.returncode})
        exit_code = exit_code or completed.returncode
    (args.output_root / "run-receipt.json").write_text(
        json.dumps({"schema_version": 1, "runs": receipts}, indent=2) + "\n",
        encoding="utf-8",
    )
    if exit_code:
        return exit_code
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("report.py")),
            str(args.output_root),
            "--parity-atol",
            str(args.parity_atol),
            "--parity-rtol",
            str(args.parity_rtol),
            "--max-relative-l2",
            str(args.max_relative_l2),
        ]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
