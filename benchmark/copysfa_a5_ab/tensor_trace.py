"""CPU-side tensor comparisons for the A5 framework-chain diagnostic."""

from __future__ import annotations

import json
from pathlib import Path


def gather_logical(torch, packed, table, ids, width):
    """Resolve logical token IDs through each request's actual physical pages."""
    packed, table, ids = (x.detach().cpu() for x in (packed, table, ids))
    ids = ids.long().reshape(table.shape[0] * width, -1)
    routes = torch.arange(ids.shape[0]) // width
    live = ids >= 0
    safe = ids.clamp_min(0)
    if bool((safe // 128 >= table.shape[1]).any()):
        raise ValueError("logical token exceeds the supplied page table")
    physical = table.long()[routes[:, None], safe // 128] * 128 + safe % 128
    flat = packed.contiguous().view(-1, 656)
    if bool(((physical < 0) | (physical >= flat.shape[0])).any()):
        raise ValueError("page table points outside packed KV")
    result = flat[physical]
    result[~live] = 0
    return result


def compare(torch, actual, expected, *, exact=True, atol=0.005, rtol=0.02, l2=0.08):
    a, b = (x.detach().cpu().contiguous() for x in (actual, expected))
    result = {
        "actual_shape": list(a.shape),
        "expected_shape": list(b.shape),
        "actual_dtype": str(a.dtype),
        "expected_dtype": str(b.dtype),
        "actual_stride": list(actual.stride()),
        "expected_stride": list(expected.stride()),
        "exact": exact,
    }
    if a.shape != b.shape or a.dtype != b.dtype:
        return result | {"passed": False, "reason": "shape or dtype mismatch"}
    af, bf = a.double(), b.double()
    finite = bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
    error = (af - bf).abs()
    mismatch = a != b if exact else error > atol + rtol * bf.abs()
    mismatch |= ~torch.isfinite(af) | ~torch.isfinite(bf)
    relative_l2 = float((af - bf).norm() / bf.norm().clamp_min(1e-12))
    result.update(
        passed=finite and not bool(mismatch.any()) and (exact or relative_l2 <= l2),
        finite=finite,
        mismatch_elements=int(mismatch.sum()),
        first_mismatch_indices=mismatch.nonzero()[:8].tolist(),
        max_abs=float(error.max()) if finite and a.numel() else None,
        relative_l2=relative_l2 if finite else None,
    )
    result["mismatch_samples"] = [
        {
            "index": index.tolist(),
            "actual": str(a[tuple(index.tolist())].item()),
            "expected": str(b[tuple(index.tolist())].item()),
        }
        for index in mismatch.nonzero()[:8]
    ]
    if not exact:
        result.update(atol=atol, rtol=rtol, max_relative_l2=l2)
    return result


class TensorTrace:
    """Persist comparisons immediately, including partial results on failure."""

    def __init__(self, torch, output: Path, metadata):
        self.torch, self.output = torch, output
        self.report = {"schema_version": 1, "metadata": metadata, "phases": {}}
        self.snapshots = {}
        self.phase = None
        self.stage = "setup"

    def begin(self, phase):
        self.phase = phase
        self.snapshots = {}
        self.report["phases"][phase] = {"checks": [], "observations": {}}

    def save(self, name, value):
        self.snapshots[name] = value.detach().cpu().clone()

    def observe(self, name, value):
        self.report["phases"][self.phase]["observations"][name] = value

    def check(self, name, actual, expected, **kwargs):
        self.stage = name
        self.save(name + ".actual", actual)
        self.save(name + ".expected", expected)
        result = compare(self.torch, actual, expected, **kwargs)
        self.report["phases"][self.phase]["checks"].append({"name": name, **result})
        self.write()
        return result["passed"]

    def require(self, condition, message):
        if not condition:
            raise ValueError(f"unsafe input at {self.stage}: {message}")

    def write(self):
        checks = [c for p in self.report["phases"].values() for c in p["checks"]]
        self.report["first_mismatch"] = next(
            (
                {"phase": phase, "check": c["name"]}
                for phase, p in self.report["phases"].items()
                for c in p["checks"]
                if not c["passed"]
            ),
            None,
        )
        self.report["passed"] = (
            self.report.get("complete", False)
            and bool(checks)
            and all(c["passed"] for c in checks)
            and "error" not in self.report
        )
        (self.output / "report.json").write_text(
            json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def flush(self):
        if self.phase is not None:
            self.torch.save(self.snapshots, self.output / f"{self.phase}.pt")
        self.write()


def summarize(report):
    """Render a short parsed result; raw tensors remain in snapshot files."""
    lines = [f"A5 C8 alignment: {'PASS' if report.get('passed') else 'FAIL'}"]
    args = report.get("metadata", {}).get("args", {})
    if args:
        lines.append(
            f"Q{args['query_width']}, batch={args['batch']}, heads={args['heads']}, "
            f"context={args['context']}, C={args['capacity']}, device={args['device']}"
        )
    for name, phase in report.get("phases", {}).items():
        checks = phase["checks"]
        passed = sum(c["passed"] for c in checks)
        observations = phase["observations"]
        lines.append(
            f"{name}: {passed}/{len(checks)} checks passed; "
            f"path={observations.get('path', 'incomplete')}; "
            f"union misses={observations.get('miss_counts', 'not used')}"
        )
    first = report.get("first_mismatch")
    if first:
        check = next(
            c
            for c in report["phases"][first["phase"]]["checks"]
            if c["name"] == first["check"]
        )
        lines.append(f"First mismatch: {first['phase']} / {first['check']}")
        lines.append(
            f"  elements={check.get('mismatch_elements', 'n/a')}, "
            f"max_abs={check.get('max_abs', 'n/a')}, rel_L2={check.get('relative_l2', 'n/a')}"
        )
        if check.get("mismatch_samples"):
            sample = check["mismatch_samples"][0]
            lines.append(
                f"  index={sample['index']}: actual={sample['actual']}, expected={sample['expected']}"
            )
    if "error" in report:
        lines.append(f"Stopped at: {report['error']['stage']} (see run.log)")
    if report.get("passed"):
        lines.append(
            "TopK, KV movement and attention outputs agree within the recorded gates."
        )
    return "\n".join(lines) + "\n"
