"""Compare separately installed LEGACY and REFACTOR wheels on identical inputs.

baseline saves inputs, results and five timing rounds; refactor reuses that
file and compares outputs directly. Neither phase changes the installed wheel.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
WARMUP, ITERATIONS, REPEATS = 5, 50, 5


def native_reference():
    path = (
        ROOT
        / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/torch_native/sampler/apply_token_bitmask.py"
    )
    spec = importlib.util.spec_from_file_location("bitmask_native_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_token_bitmask


def inputs(shape, dtype, mode, generator, indices=None):
    logits = torch.randn(shape, generator=generator).to(dtype)
    width = (shape[1] + 31) // 32
    if mode == "random":
        mask = torch.randint(
            -(2**31), 2**31, (shape[0], width), generator=generator, dtype=torch.int32
        )
    else:
        mask = torch.full(
            (shape[0], width),
            {"all_masked": 0, "all_unmasked": -1, "half_masked": 0x55555555}[mode],
            dtype=torch.int32,
        )
    return {"logits": logits, "bitmask": mask, "indices": indices}


def cases():
    generator = torch.Generator().manual_seed(20261022)
    accuracy = []
    for shape in [(1, 64), (4, 320), (64, 32000), (2, 128256)]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for mode in ["all_masked", "all_unmasked", "half_masked", "random"]:
                accuracy.append(inputs(shape, dtype, mode, generator))
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for shape, indices in [
            ((8, 320), [0, 2, 5, 7]),
            ((4, 320), list(range(4))),
            ((4, 320), []),
        ]:
            accuracy.append(
                inputs(
                    shape,
                    dtype,
                    "random",
                    generator,
                    torch.tensor(indices, dtype=torch.int32),
                )
            )
    performance = []
    for batch, vocab, dtype in [
        (16, 32000, torch.float16),
        (64, 32000, torch.float16),
        (1, 128256, torch.float16),
        (4, 128256, torch.float16),
        (16, 32000, torch.bfloat16),
        (16, 32000, torch.float32),
        (1, 320, torch.float16),
    ]:
        performance.append(inputs((batch, vocab), dtype, "random", generator))
    for batch, vocab in [(16, 32000), (64, 32000)]:
        indices = torch.randperm(batch, generator=generator, dtype=torch.int32)[
            : batch // 2
        ]
        performance.append(
            inputs((batch, vocab), torch.float16, "random", generator, indices)
        )
    return accuracy, performance


def compare(actual, expected):
    assert torch.equal(
        torch.isneginf(actual), torch.isneginf(expected)
    ), "-inf positions differ"
    finite = ~torch.isneginf(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=1e-2, rtol=1e-2)


def accuracy_case(operator, case, reference):
    values = case["logits"].npu()
    bits = case["bitmask"].npu()
    rows = None if case["indices"] is None else case["indices"].npu()
    result = operator(values, bits, rows)
    assert result.data_ptr() == values.data_ptr(), "Return does not alias logits"
    actual = result.cpu()
    expected = reference(
        case["logits"].clone(), case["bitmask"].clone(), case["indices"]
    )
    compare(actual, expected)
    assert torch.equal(
        actual[~torch.isneginf(expected)], case["logits"][~torch.isneginf(expected)]
    )
    assert torch.equal(bits.cpu(), case["bitmask"])
    if rows is not None:
        assert torch.equal(rows.cpu(), case["indices"])
    return actual


def time_case(operator, case):
    base = case["logits"].npu()
    bits = case["bitmask"].npu()
    rows = None if case["indices"] is None else case["indices"].npu()
    timings = []
    for _ in range(REPEATS):
        # All calls receive the same original values. Preparation is untimed.
        values = [base.clone() for _ in range(WARMUP + ITERATIONS)]
        torch.npu.synchronize()
        for value in values[:WARMUP]:
            operator(value, bits, rows)
        torch.npu.synchronize()
        measured = values[WARMUP:]
        start = time.perf_counter()
        for value in measured:
            operator(value, bits, rows)
        torch.npu.synchronize()
        timings.append((time.perf_counter() - start) * 1000 / ITERATIONS)
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "refactor"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import sgl_kernel_npu
    import torch_npu

    package = Path(sgl_kernel_npu.__file__).parent
    manifest = package / "_build_manifest.json"
    if (args.phase == "refactor") != manifest.exists():
        raise RuntimeError("Installed wheel does not match the requested phase")
    reference = native_reference()
    if args.phase == "baseline":
        if args.baseline.exists():
            raise RuntimeError(
                "Baseline already exists; select a new path to preserve the previous measurement"
            )
        accuracy, performance = cases()
    else:
        baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
        accuracy, performance = (
            baseline["accuracy_inputs"],
            baseline["performance_inputs"],
        )
    report = {
        "phase": args.phase,
        "required_baseline_commit": "ecd3f947e9ed6cd512225c5030f2fd6deb63233c",
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "package_version": importlib.metadata.version("sgl_kernel_npu"),
        "device": torch.npu.current_device(),
        "device_name": torch.npu.get_device_name(),
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "input_preparation_timed": False,
    }
    library = package / "lib/libsgl_kernel_npu.so"
    report["native_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    sdk = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    report["cann_version_info"] = {
        name: (sdk / name).read_text()
        for name in ("version.cfg", "version.info", "compiler/version.info")
        if (sdk / name).is_file()
    }
    if args.phase == "refactor":
        for key in (
            "torch",
            "torch_npu",
            "device",
            "device_name",
            "cann_version_info",
            "warmup",
            "iterations",
            "repeats",
        ):
            if report[key] != baseline["report"][key]:
                raise RuntimeError(
                    f"Baseline environment/protocol changed: {key}; preserve it before comparing"
                )
    if manifest.exists():
        report["manifest"] = json.loads(manifest.read_text())
    direct = torch.ops.npu.apply_token_bitmask
    outputs = [accuracy_case(direct, case, reference) for case in accuracy]
    direct_ms = [time_case(direct, case) for case in performance]
    report["accuracy_cases"] = len(outputs)
    report["direct_ms"] = direct_ms
    if args.phase == "baseline":
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "accuracy_inputs": accuracy,
                "performance_inputs": performance,
                "outputs": outputs,
                "report": report,
            },
            args.baseline,
        )
    else:
        for actual, expected in zip(outputs, baseline["outputs"], strict=True):
            compare(actual, expected)
        from sgl_kernel_npu.operators.sampler import apply_token_bitmask

        for case, expected in zip(accuracy, outputs, strict=True):
            compare(accuracy_case(apply_token_bitmask, case, reference), expected)
        public_ms = [time_case(apply_token_bitmask, case) for case in performance]
        report["operators_ms"] = public_ms
        report["comparison"] = []
        for old, new in zip(baseline["report"]["direct_ms"], public_ms, strict=True):
            old_median, new_median = statistics.median(old), statistics.median(new)
            overlap = max(min(old), min(new)) <= min(max(old), max(new))
            report["comparison"].append(
                {
                    "legacy_median_ms": old_median,
                    "refactor_operators_median_ms": new_median,
                    "ratio": new_median / old_median,
                    "status": (
                        "requires_retest"
                        if overlap or new_median > old_median
                        else "no_regression_observed"
                    ),
                }
            )
        report["output_comparison"] = "passed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
