#!/usr/bin/env python3
"""Prepare the pinned, unmodified A5 LIM OPP sources."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def prepare(output_root: Path) -> Path:
    repo = Path(__file__).resolve().parents[2]
    helper = repo / "benchmark/lim_a5_mtp_ab/prepare_original.py"
    spec = importlib.util.spec_from_file_location("_sgl_original_source", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_path = module.prepare(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["operators"] = {
        "lim": {"revision": module.UPSTREAM_REVISION, "pr": 18383},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(prepare(arguments.output_root))
