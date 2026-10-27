"""Execute a validated selection, preserving legacy script and pytest semantics."""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def run(plan, module, root, cann):
    root = root.resolve()
    config = plan["modules"][module]
    tests = config["tests"]
    reports = root / "test-results"
    reports.mkdir(exist_ok=True)
    # Validate the entire list before running anything. A stale/malformed plan is
    # a failure, never an accidental full pytest collection or a silent skip.
    for test in tests:
        path = (root / test).resolve()
        if (
            not path.is_relative_to(root / "tests/python")
            or not path.is_file()
            or path.suffix != ".py"
            or not path.name.startswith("test_")
        ):
            raise ValueError(f"Invalid or missing selected test: {test}")
    print(f"Selected {module}, CANN {cann}: {json.dumps(tests)}", flush=True)
    skipped = plan["cann_exclusions"].get(cann, {})
    public = [t for t in tests if t.startswith("tests/python/operators/")]
    legacy = [t for t in tests if t not in public]
    failures = []
    env = dict(os.environ, ACL_OP_SELECT_IMPL_MODE="high_precision")
    # Finish all collected cases, even when defaults in PYTEST_ADDOPTS use -x.
    # Collection errors still fail the run but do not suppress valid tests.
    if public:
        code = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *public,
                "--impl=triton",
                "--impl-ref=torch_native",
                "--skip-acc-check=false",
                "--enable-perf=false",
                "--npu-device=0",
                "-v",
                "-rs",
                "--maxfail=0",
                "--continue-on-collection-errors",
                f"--junitxml={reports / ('public-' + module + '.xml')}",
            ],
            cwd=root,
            env=env,
        ).returncode
        if code:
            failures.append(f"public {module} (exit {code})")
    for test in legacy:
        path = root / test
        report = reports / ("legacy-" + path.stem + ".xml")
        if path.name in skipped:
            reason = skipped[path.name]
            print(f"SKIP {test} on CANN {cann}: {reason}", flush=True)
            suite = ET.Element("testsuite", name=test, tests="1", skipped="1")
            ET.SubElement(
                ET.SubElement(suite, "testcase", name=test), "skipped", message=reason
            )
            ET.ElementTree(suite).write(report, encoding="utf-8", xml_declaration=True)
            continue
        source = path.read_text()
        script = "__main__" in source or "def test_" not in source
        command = (
            [sys.executable, path.name]
            if script
            else [
                sys.executable,
                "-m",
                "pytest",
                path.name,
                "-q",
                "--maxfail=0",
                "--continue-on-collection-errors",
                f"--junitxml={report}",
            ]
        )
        print(f"RUN {test}", flush=True)
        started = time.monotonic()
        code = subprocess.run(command, cwd=path.parent, env=env).returncode
        if script:
            suite = ET.Element(
                "testsuite", name=test, tests="1", failures=str(int(code != 0))
            )
            case = ET.SubElement(
                suite, "testcase", name=test, time=str(time.monotonic() - started)
            )
            if code:
                ET.SubElement(
                    case, "failure", message=f"Script exit status {code}; see job log"
                )
            ET.ElementTree(suite).write(report, encoding="utf-8", xml_declaration=True)
        if code:
            failures.append(f"{test} (exit {code})")
    print(f"{module}: {len(tests)} selected, failures: {failures}", flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(f"## {module}, CANN {cann}\n\n")
            output.write("\n".join(f"- `{t}`" for t in tests) + "\n")
            output.write(f"\nFailures: {failures or 'none'}\n")
            output.write(
                "\n".join(
                    f"- Existing skip: {Path(t).name}: {skipped[Path(t).name]}"
                    for t in legacy
                    if Path(t).name in skipped
                )
                + "\n"
            )
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("module")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    return run(
        json.loads(args.selection.read_text()),
        args.module,
        root,
        os.environ.get("CANN_VERSION", ""),
    )


if __name__ == "__main__":
    sys.exit(main())
