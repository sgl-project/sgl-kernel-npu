"""Collect Python operator coverage from the repository's pytest suites."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS = ("tests/python/operators", "tests/python/framework")
SOURCE_PATTERNS = "*/sgl_kernel_npu/implementation/*,*/sgl_kernel_npu/operators/*"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", help="Test paths relative to the repo root")
    parser.add_argument("--output-dir", type=Path, default=Path("coverage-out"))
    parser.add_argument("--no-merge", action="store_true", help="Collect only; runner merges after all files")
    parser.add_argument("--merge-only", action="store_true", help="Merge existing successful per-file data")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "coverage.xml"
    report.unlink(missing_ok=True)
    combined_data = output_dir / ".coverage"
    combined_data.unlink(missing_ok=True)
    env = {**os.environ, "COVERAGE_FILE": str(combined_data)}
    coverage = [sys.executable, "-m", "coverage"]

    test_files = set()
    for test in ([] if args.merge_only else args.tests or DEFAULT_TESTS):
        path = (ROOT / test).resolve()
        if not path.is_relative_to(ROOT) or not path.exists():
            parser.error(
                f"Expected an existing test path inside the repository: {test}"
            )
        test_files.update(path.rglob("test_*.py") if path.is_dir() else [path])
    if not test_files and not args.merge_only:
        parser.error("No test files found")

    data_files = []
    failed = False
    for test_file in sorted(test_files):
        relative_path = test_file.relative_to(ROOT)
        data_dir = output_dir / relative_path / "covdata"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "coverage"
        lines_report = data_dir / "coverage.json"
        failure_marker = data_dir / "FAILED"
        data_file.unlink(missing_ok=True)
        lines_report.unlink(missing_ok=True)
        failure_marker.unlink(missing_ok=True)
        print(f"Collecting coverage for {relative_path}", flush=True)
        test_env = {**env, "COVERAGE_FILE": str(data_file)}
        result = subprocess.run(
            [
                *coverage,
                "run",
                "--branch",
                f"--include={SOURCE_PATTERNS}",
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--enable-perf=false",
                str(relative_path),
            ],
            cwd=ROOT,
            env=test_env,
        )
        if result.returncode or not data_file.is_file():
            failure_marker.write_text(f"pytest exit code: {result.returncode}\n")
            failed = True
            if not data_file.is_file():
                continue

        export = subprocess.run(
            [*coverage, "json", "-o", str(lines_report)],
            cwd=ROOT,
            env=test_env,
        )
        if export.returncode:
            failure_marker.write_text(f"Coverage JSON exit code: {export.returncode}\n")
            failed = True
            continue
        data_files.append(str(data_file))

    if args.no_merge:
        return int(failed)
    if args.merge_only:
        data_files = [str(p) for p in sorted(output_dir.glob("tests/**/covdata/coverage"))
                      if (p.parent / "coverage.json").is_file()]
        failed_files = [p.parent.parent.relative_to(output_dir).as_posix()
                        for p in sorted(output_dir.glob("tests/**/covdata/FAILED"))]
        manifest = {
            "hardware": os.environ.get("NPU_HARDWARE", ""),
            "test_files_with_coverage": len(data_files),
            "failed_test_files": failed_files,
            "incomplete": bool(failed_files),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    if not data_files:
        return 1

    combined = subprocess.run(
        [*coverage, "combine", "--keep", *data_files],
        cwd=ROOT,
        env=env,
    )
    if combined.returncode:
        return combined.returncode

    report_result = subprocess.run(
        [*coverage, "xml", "-o", str(report)],
        cwd=ROOT,
        env=env,
    )
    return report_result.returncode or int(failed)


if __name__ == "__main__":
    sys.exit(main())
