#!/usr/bin/env python3
"""Build/install/test the native selection matrix; continue after scenario failures.

Run on the matching NPU after sourcing CANN:
  python3 scripts/validate_native_builds.py --arch-version 2201 --jobs 8
  python3 scripts/validate_native_builds.py --arch-version 3510 --jobs 8
Preview without an SDK/NPU:
  python3 scripts/validate_native_builds.py --arch-version 2201 --list

Results include JSON/CSV/Markdown summaries, commands, build logs, installers,
wheels, installed manifests and per-file JUnit XML. No operator tests are run
against a RUN alone: RUN checks installation/payloads, wheels check execution.
The default matrix covers execution-unit combinations and individual operators
for both artifacts, plus incremental all rebuilds. The all wheel also runs the
remaining operator and framework tests.
Third-party dependencies are excluded unless --third-party is supplied.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import venv
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Explicit mapping: a new native ID must get a test mapping before this runner
# can claim full coverage. Registration/architecture/selection come from CMake.
OPERATORS = {
    "apply_token_bitmask": (
        "sampler/test_apply_token_bitmask.py",
        "apply_token_bitmask",
    ),
    "chunk_gated_delta_rule": (
        "attention/test_chunk_gated_delta_rule.py",
        "sgl_chunk_gated_delta_rule",
    ),
    "cache_loc_assign": ("mem_cache/test_cache_locations.py", "cache_loc_assign"),
    "cache_loc_update": ("mem_cache/test_cache_locations.py", "cache_loc_update"),
    "sgemmv_expand": ("lora/test_sgemmv_expand.py", "sgemmv_expand"),
    "add_example": ("examples/test_add_example.py", "sgl_add_example"),
    "alloc_extend": ("mem_cache/test_alloc_extend.py", "alloc_extend"),
    "assign_cache_op": ("mem_cache/test_assign_cache_op.py", "assign_cache_op"),
    "batch_matmul_transpose": (
        "matmul/test_batch_matmul_transpose.py",
        "batch_matmul_transpose",
    ),
    "compressor": ("attention/test_compressor.py", "compressor"),
    "kv_compress_epilog": (
        "mem_cache/test_kv_compress_epilog.py",
        "kv_compress_epilog",
    ),
    "mla_preprocess": ("attention/test_mla_preprocess.py", "mla_preprocess"),
    "sparse_attn_sharedkv": (
        "attention/test_sparse_attn_sharedkv.py",
        "sparse_attn_sharedkv",
    ),
    "sparse_flash_attention": (
        "attention/test_sparse_flash_attention.py",
        "sparse_flash_attention",
    ),
    "swiglu_group_quant": (
        "activation/test_swiglu_group_quant.py",
        "swiglu_group_quant",
    ),
    "transfer_kv_dim_exchange": (
        "data_move/test_transfer_kv_dim_exchange.py",
        "transfer_kv_dim_exchange",
    ),
    "transfer_mamba_state": (
        "data_move/test_transfer_mamba_state.py",
        "transfer_mamba_state",
    ),
    "transfer_kv_per_layer_direct_pf_lf": (
        "data_move/test_transfer_state_direct.py",
        "transfer_state_per_layer_direct_pf_lf",
    ),
    "transfer_kv_all_layer_direct_lf_pf": (
        "data_move/test_transfer_state_direct.py",
        "transfer_state_all_layer_direct_lf_pf",
    ),
}
PER_LAYER = "transfer_kv_per_layer_direct_pf_lf"
ALL_LAYERS = "transfer_kv_all_layer_direct_lf_pf"

# Use the registry and declaration-local dependency expansion, without an SDK
# or compilation. Keep HOST IDs away from legacy CANN declaration discovery.
DISCOVERY = r"""
cmake_minimum_required(VERSION 3.20)
project(sgl_build_validation LANGUAGES NONE)
include("${REPOSITORY}/csrc/cmake/operators.cmake")
include("${REPOSITORY}/csrc/cmake/domains.cmake")
sgl_add_host_operator_domains()
sgl_get_registered_operators(registered_host EXECUTION_UNIT HOST)
set(cann_request)
set(host_request)
foreach(id IN LISTS REQUESTED)
    if(id IN_LIST registered_host)
        list(APPEND host_request "${id}")
    else()
        list(APPEND cann_request "${id}")
    endif()
endforeach()
set(SGL_CANN_OPERATORS "${cann_request}" CACHE STRING "CANN selection" FORCE)
sgl_add_cann_operator_domains()
sgl_select_native_operators(cann host ARCH "${ARCH}"
    OPERATORS ${SGL_CANN_OPERATORS} ${host_request})
file(WRITE "${CMAKE_BINARY_DIR}/selection.tsv" "")
foreach(id IN LISTS cann host)
    sgl_get_operator_field(unit "${id}" EXECUTION_UNIT)
    file(APPEND "${CMAKE_BINARY_DIR}/selection.tsv" "${id}\t${unit}\n")
endforeach()
"""


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def discover(directory, arch, requested):
    source = directory / "source"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(DISCOVERY)
    build = directory / "build"
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        f"-DREPOSITORY={ROOT}",
        f"-DARCH={arch}",
        "-DREQUESTED="
        + ("ALL" if requested in (None, "all") else requested.replace(",", ";")),
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    (directory / "configure.log").write_text(shlex.join(command) + "\n" + result.stdout)
    if result.returncode:
        raise RuntimeError(f"CMake selection failed; see {directory / 'configure.log'}")
    return dict(
        line.split("\t") for line in (build / "selection.tsv").read_text().splitlines()
    )


def test_groups(selected):
    groups = {}
    for operator, unit in selected.items():
        path = "tests/python/operators/" + OPERATORS[operator][0]
        backend = {"HOST": "aclrt", "AICPU": "aicpu", "AICORE": "ascendc"}[unit]
        groups[path] = {"file": path, "impl": backend, "required": True, "filter": None}
    path = "tests/python/operators/data_move/test_transfer_state_direct.py"
    if path in groups and not {PER_LAYER, ALL_LAYERS}.issubset(selected):
        # Shared tests contain both one-way copies and round trips. Single-ID
        # wheels must exercise only the API actually selected for that wheel.
        groups[path]["filter"] = (
            "single-layer-duplicate-source or mapping-order or empty or transpose-h2d "
            "or multi-transpose-h2d or test_state_transfer_rejected "
            "or (test_state_transfer_repeated_destination and H2D)"
            if PER_LAYER in selected
            else "all-layers-scattered or transpose-d2h or multi-transpose-d2h "
            "or (test_state_transfer_repeated_destination and D2H)"
        )
    return [groups[path] for path in sorted(groups)]


def include_framework_tests(scenarios, native_only):
    if native_only:
        return
    native_files = {
        "tests/python/operators/" + value[0] for value in OPERATORS.values()
    }
    extra_files = sorted((ROOT / "tests/python/operators").rglob("test_*.py"))
    extra_files += sorted((ROOT / "tests/python/framework").rglob("test_*.py"))
    for scenario in scenarios:
        if scenario["name"] == "all":
            scenario["tests"] += [
                {
                    "file": str(p.relative_to(ROOT)),
                    "impl": "auto",
                    "required": False,
                    "filter": None,
                }
                for p in extra_files
                if str(p.relative_to(ROOT)) not in native_files
            ]


def matrix(directory, arch, scope):
    inventory = discover(directory / "inventory", arch, "all")
    missing = set(inventory).difference(OPERATORS)
    if missing:
        raise RuntimeError(
            "Add test/schema mappings for native IDs: " + ", ".join(sorted(missing))
        )
    for operator in inventory:
        path = ROOT / "tests/python/operators" / OPERATORS[operator][0]
        if not path.is_file():
            raise RuntimeError(f"Missing mapped test: {path}")
    units = {
        unit: sorted(op for op, kind in inventory.items() if kind == unit)
        for unit in ("AICORE", "AICPU", "HOST")
    }
    requests = []
    for name, kinds in [
        ("host-only", ["HOST"]),
        ("aicore-only", ["AICORE"]),
        ("aicpu-only", ["AICPU"]),
        ("cann-only", ["AICORE", "AICPU"]),
        ("host-aicore", ["HOST", "AICORE"]),
        ("host-aicpu", ["HOST", "AICPU"]),
    ]:
        if all(units[kind] for kind in kinds):
            requests.append(
                (name, ",".join(op for kind in kinds for op in units[kind]))
            )
    requests += [
        ("all", "all"),
        ("mixed-explicit", ",".join(sorted(inventory))),
        ("default-ops", None),
    ]
    if scope == "full":
        requests += [("single-" + operator, operator) for operator in sorted(inventory)]
    scenarios = []
    for name, requested in requests:
        selected = discover(directory / name, arch, requested)
        scenarios.append(
            {
                "name": name,
                "ops": requested,
                "selected": selected,
                "tests": test_groups(selected),
            }
        )
    return inventory, scenarios


def junit_counts(path):
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    counts = dict(passed=0, failed=0, errors=0, skipped=0)
    reasons = []
    for case in root.iter("testcase"):
        category = "passed"
        for tag, kind in (
            ("failure", "failed"),
            ("error", "errors"),
            ("skipped", "skipped"),
        ):
            node = case.find(tag)
            if node is not None:
                category = kind
                reasons.append(
                    {
                        "test": case.get("name"),
                        "status": kind,
                        "reason": node.get("message") or (node.text or "")[:500],
                    }
                )
                break
        counts[category] += 1
    return {**counts, "details": reasons}


class Runner:
    def __init__(self, output):
        self.output = output
        self.cases = []
        self.context = {}

    def save(self):
        write_json(self.output / "results.json", {**self.context, "cases": self.cases})
        with (self.output / "results.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "scenario",
                    "artifact",
                    "status",
                    "stage",
                    "exit_code",
                    "seconds",
                    "passed",
                    "failed",
                    "errors",
                    "skipped",
                    "log",
                ]
            )
            for case in self.cases:
                for step in case["steps"]:
                    counts = step.get("tests") or {}
                    writer.writerow(
                        [
                            case["name"],
                            case["artifact"],
                            case["status"],
                            step["name"],
                            step["exit_code"],
                            step["seconds"],
                            *(
                                counts.get(key, "")
                                for key in ("passed", "failed", "errors", "skipped")
                            ),
                            step["log"],
                        ]
                    )
        lines = [
            "# Native build validation",
            "",
            f"Architecture: {self.context.get('arch_version', '')}; Git: `{self.context.get('git', '')}`",
            f"Overall exit code: {self.context.get('exit_code', 'running')}",
            "",
            "RUN checks installation/payloads. Operator execution is checked by the corresponding wheels.",
            "",
            "| Scenario | Artifact | Status | Passed | Failed/errors | Skipped | Reason |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for case in self.cases:
            totals = {
                key: sum(
                    (step.get("tests") or {}).get(key, 0) for step in case["steps"]
                )
                for key in ("passed", "failed", "errors", "skipped")
            }
            lines.append(
                f"| {case['name']} | {case['artifact']} | {case['status']} | {totals['passed']} | {totals['failed'] + totals['errors']} | {totals['skipped']} | {case.get('reason', '').replace('|', '/').replace(chr(10), ' ')} |"
            )
        lines += [
            "",
            "Detailed commands, reasons, per-test skips and exit codes: results.json / results.csv and each scenario's logs/JUnit XML.",
            "",
        ]
        (self.output / "SUMMARY.md").write_text("\n".join(lines))

    def step(self, case, name, command, *, cwd, timeout=0, env=None):
        folder = self.output / case["name"] / case["artifact"]
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / (name + ".log")
        print(
            f"[{case['name']}/{case['artifact']}] {name}\n  $ {shlex.join(map(str, command))}\n  log: {log}",
            flush=True,
        )
        start = time.monotonic()
        result = {
            "name": name,
            "command": list(map(str, command)),
            "log": str(log.relative_to(self.output)),
            "exit_code": 1,
        }
        with log.open("w") as stream:
            stream.write(shlex.join(map(str, command)) + "\n")
            stream.flush()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
                try:
                    code = process.wait(timeout=timeout or None)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    if isinstance(error, KeyboardInterrupt):
                        result["exit_code"] = 130
                        raise
                    stream.write(f"\nTIMEOUT after {timeout}s\n")
                    code = 124
                result["exit_code"] = code
            except KeyboardInterrupt:
                result["exit_code"] = 130
                raise
            except OSError as error:
                stream.write(str(error) + "\n")
                result["exit_code"] = 127
            finally:
                result["seconds"] = round(time.monotonic() - start, 2)
                case["steps"].append(result)
                if result.get("exit_code", 1):
                    case["status"] = "FAIL"
                self.save()
        print(f"  exit={result['exit_code']} elapsed={result['seconds']}s", flush=True)
        return result


def verify_manifest(manifest, spec):
    selected = spec["selected"]
    cann = sorted(op for op, unit in selected.items() if unit != "HOST")
    host = sorted(op for op, unit in selected.items() if unit == "HOST")
    expected = {
        "arch_version": spec["arch_version"],
        "selection_kind": spec["selection_kind"],
        "operators": sorted(selected),
        "operators_by_type": {"cann": cann, "cpp": host},
        "compute_units": spec["compute_units"] if cann else [],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Manifest mismatch for {key}: expected {value!r}, got {manifest.get(key)!r}"
            )
    if not spec["third_party"] and manifest.get("third_party"):
        raise RuntimeError(
            "Unexpected third-party payload in --without-third-party build"
        )


def check_wheel(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    with zipfile.ZipFile(spec["wheel"]) as archive:
        libraries = {
            name
            for name in archive.namelist()
            if name.startswith("sgl_kernel_npu/lib/") and name.endswith(".so")
        }
        if libraries != {"sgl_kernel_npu/lib/libsgl_kernel_npu.so"}:
            raise RuntimeError(
                f"Expected one Torch operator library, got {sorted(libraries)}"
            )
        manifest_bytes = archive.read("sgl_kernel_npu/_build_manifest.json")
        library_bytes = archive.read("sgl_kernel_npu/lib/libsgl_kernel_npu.so")
    manifest = json.loads(manifest_bytes)
    verify_manifest(manifest, spec)
    # Import package first so its vendor paths are enabled before torch_npu.
    import sgl_kernel_npu
    import torch

    package = Path(sgl_kernel_npu.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"Imported package outside isolated test environment: {package}"
        )
    if (package / "_build_manifest.json").read_bytes() != manifest_bytes:
        raise RuntimeError("Installed manifest differs from the wheel just built")
    if (
        hashlib.sha256((package / "lib/libsgl_kernel_npu.so").read_bytes()).digest()
        != hashlib.sha256(library_bytes).digest()
    ):
        raise RuntimeError("Installed Torch library differs from the wheel just built")
    schemas = {}
    for operator, (_, schema) in OPERATORS.items():
        present = hasattr(torch.ops.npu, schema)
        schemas[operator] = present
        if present != (operator in spec["selected"]):
            raise RuntimeError(
                f"Wrong Torch registration: {operator} ({schema}), present={present}"
            )
        if present and not torch._C._dispatch_has_kernel_for_dispatch_key(
            "npu::" + schema, "PrivateUse1"
        ):
            raise RuntimeError(f"Missing NPU dispatch kernel: npu::{schema}")
    vendor = package / "_cann/vendors/sgl_kernel_npu"
    if bool(manifest["operators_by_type"]["cann"]) != vendor.is_dir():
        raise RuntimeError("Repository CANN vendor presence disagrees with selection")
    from sgl_kernel_npu.operators.utils.generation import get_hardware

    hardware = get_hardware()
    if str(hardware.arch_id) != spec["arch_version"]:
        raise RuntimeError(
            f"Runtime NPU architecture {hardware.arch_id} differs from requested {spec['arch_version']}"
        )
    write_json(Path(spec_path).with_name("installed-manifest.json"), manifest)
    write_json(
        Path(spec_path).with_name("runtime.json"),
        {
            "python": sys.executable,
            "package": str(package),
            "torch": torch.__version__,
            "schemas": schemas,
            "arch": hardware.arch_id,
        },
    )
    print(
        "PASS: exact wheel installation, manifest, selection and Torch NPU registrations"
    )


def check_run(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    installed = Path(spec["installed"])
    # Reuse the actual build's artifact verifier for CANN libraries, Kernel
    # binaries per compute unit, AICPU config/library, and the HOST archive.
    sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
    from build_tools.cann import inspect_installation

    units = set(spec["selected"].values())
    vendor, libraries, payload = inspect_installation(
        installed,
        ROOT / "build/cann",
        spec["compute_units"],
        aicore="AICORE" in units,
        aicpu="AICPU" in units,
        host="HOST" in units,
    )
    archive = vendor / "op_impl/host/lib/libsgl_host_operators.a"
    if archive.is_file() != any(unit == "HOST" for unit in spec["selected"].values()):
        raise RuntimeError("HOST archive presence disagrees with selection")
    write_json(
        Path(spec_path).with_name("installed-payload.json"),
        {
            "vendor": str(vendor),
            "files": [str(p.relative_to(vendor)) for p in libraries + payload],
        },
    )
    print("PASS: RUN executed and installed artifacts match the verified build")


def create_environment(directory):
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(directory)
    python = directory / "bin/python"
    # A nested venv does not inherit its parent's site-packages automatically.
    # Retain dependency paths (Torch/CANN/pytest), excluding checkout/stage paths.
    paths = []
    for entry in sys.path:
        path = Path(entry or os.getcwd()).resolve()
        if (
            path.is_dir()
            and path not in (ROOT, ROOT / "scripts", ROOT / "python/sgl_kernel_npu")
            and not path.is_relative_to(ROOT / "build")
        ):
            paths.append(str(path))
    purelib = subprocess.check_output(
        [
            str(python),
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        text=True,
    ).strip()
    (Path(purelib) / "validation-dependencies.pth").write_text(
        "\n".join(dict.fromkeys(paths)) + "\n"
    )
    return python


def preserve_build_logs(folder):
    for kind in ("cann", "torch", "wheel"):
        source = ROOT / "build" / kind
        if source.is_dir():
            for pattern in ("*.log", "*.txt", "torch-environment.json"):
                for file in source.glob(pattern):
                    if file.is_file():
                        destination = folder / "build-logs" / kind / file.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, destination)


def verify_selection(selected):
    for name, unit in (
        ("selected-operators.txt", "cann"),
        ("selected-host-operators.txt", "host"),
    ):
        actual = set((ROOT / "build/cann" / name).read_text().split())
        expected = {
            op for op, kind in selected.items() if (kind == "HOST") == (unit == "host")
        }
        if actual != expected:
            raise RuntimeError(
                f"CMake selection mismatch in {name}: expected {expected}, got {actual}"
            )


def execute_case(runner, case, scenario, args, python):
    folder = runner.output / case["name"] / case["artifact"]
    folder.mkdir(parents=True, exist_ok=True)
    selected = scenario["selected"]
    spec = {
        "arch_version": args.arch_version,
        "selected": selected,
        "selection_kind": "all" if scenario["ops"] in (None, "all") else "custom",
        "compute_units": json.loads(
            (ROOT / "csrc/cmake/architectures.json").read_text()
        )[args.arch_version]["compute_units"],
        "third_party": bool(args.third_party),
        "tests": scenario["tests"],
    }
    spec_path = folder / "spec.json"
    write_json(spec_path, spec)
    command = [
        str(python),
        "-I",
        str(ROOT / "setup.py"),
        "--arch-version",
        args.arch_version,
        "--artifact",
        case["artifact"],
        "--make-clean",
    ]
    if scenario["ops"] is not None:
        command += ["--ops", scenario["ops"]]
    if args.jobs:
        command += ["--jobs", str(args.jobs)]
    if case["artifact"] == "wheel":
        command += (
            ["--third-party", args.third_party]
            if args.third_party
            else ["--without-third-party"]
        )
    if runner.step(case, "build", command, cwd=ROOT, timeout=args.build_timeout)[
        "exit_code"
    ]:
        case["reason"] = "Build failed; install/runtime tests skipped"
        return
    verify_selection(selected)
    if scenario["name"] == "all" and not args.skip_incremental:
        incremental = [part for part in command if part != "--make-clean"]
        if runner.step(
            case, "incremental-build", incremental, cwd=ROOT, timeout=args.build_timeout
        )["exit_code"]:
            case["reason"] = "Same-selection incremental rebuild failed"
            return
        verify_selection(selected)
    suffix = "*.whl" if case["artifact"] == "wheel" else "*.run"
    artifacts = list((ROOT / "output").glob("sgl_kernel_npu-" + suffix))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected exactly one new artifact, got {artifacts}")
    artifact = folder / artifacts[0].name
    shutil.copy2(artifacts[0], artifact)
    case["output"] = str(artifact.relative_to(runner.output))
    if case["artifact"] == "run":
        spec["installed"] = str(folder / "installed")
        write_json(spec_path, spec)
        install = [
            "bash",
            str(artifact),
            "--quiet",
            "--install-path=" + spec["installed"],
        ]
        if "AICORE" not in selected.values():
            install.append("--force")
        if runner.step(
            case, "install-run", install, cwd=folder, timeout=args.build_timeout
        )["exit_code"]:
            return
        runner.step(
            case,
            "check-run",
            [
                str(python),
                "-I",
                str(Path(__file__).resolve()),
                "--_check-run",
                str(spec_path),
            ],
            cwd=folder,
        )
        case["execution"] = (
            "Installer/payload only; operator execution belongs to the matching wheel"
        )
        return
    spec["wheel"] = str(artifact)
    write_json(spec_path, spec)
    if runner.step(
        case,
        "install-wheel",
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(artifact),
        ],
        cwd=folder,
    )["exit_code"]:
        return
    if runner.step(
        case,
        "check-wheel",
        [
            str(python),
            "-I",
            str(Path(__file__).resolve()),
            "--_check-wheel",
            str(spec_path),
        ],
        cwd=folder,
    )["exit_code"]:
        case["reason"] = (
            "Installation/manifest/schema check failed; operator tests skipped"
        )
        return
    for index, group in enumerate(scenario["tests"]):
        report = folder / f"pytest-{index:03d}.xml"
        command = [
            str(python),
            "-I",
            "-m",
            "pytest",
            str(ROOT / group["file"]),
            "--impl=" + group["impl"],
            "--impl-ref=torch_native",
            "--skip-acc-check=false",
            "--enable-perf=false",
            "--npu-device=" + str(args.npu_device),
            "--maxfail=0",
            "--continue-on-collection-errors",
            # Full validation includes supplementary extra_checks too.
            "-o",
            "addopts=",
            "--junitxml=" + str(report),
            "-v",
            "-ra",
        ]
        if group["filter"]:
            command += ["-k", group["filter"]]
        environment = os.environ.copy()
        environment.pop("PYTEST_ADDOPTS", None)
        result = runner.step(
            case,
            f"pytest-{index:03d}",
            command,
            cwd=ROOT,
            timeout=args.test_timeout,
            env=environment,
        )
        result["test_file"] = group["file"]
        result["required"] = group["required"]
        result["junit"] = str(report.relative_to(runner.output))
        try:
            result["tests"] = junit_counts(report)
        except ET.ParseError as error:
            result["report_error"] = str(error)
        counts = result.get("tests")
        if (
            counts is None
            or counts["failed"]
            or counts["errors"]
            or (group["required"] and not counts["passed"])
        ):
            case["status"] = "FAIL"
            case["reason"] = (
                f"Test failures/incomplete native coverage; see pytest logs ({group['file']})"
            )
            result["coverage_error"] = (
                "Missing/failed JUnit results, or required native file has no passing tests"
            )
        runner.save()


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arch-version", choices=("2201", "3510"), required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--npu-device", type=int, default=0)
    parser.add_argument(
        "--scope",
        choices=("full", "combinations"),
        default="full",
        help="full adds individual operator selections",
    )
    parser.add_argument(
        "--artifacts", default="wheel,run", help="wheel,run (default), wheel, or run"
    )
    parser.add_argument(
        "--scenarios", help="comma-separated scenario names from --list"
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="omit non-native operator/framework tests in the all wheel",
    )
    parser.add_argument(
        "--third-party",
        help="also build these third-party profiles in wheel scenarios; default excludes third-party dependencies",
    )
    parser.add_argument(
        "--skip-incremental",
        action="store_true",
        help="omit the all scenario's same-selection incremental rebuild",
    )
    parser.add_argument(
        "--build-timeout",
        type=int,
        default=0,
        help="seconds per build/install, 0 means unlimited",
    )
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=0,
        help="seconds per test file, 0 means unlimited",
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--list",
        "--dry-run",
        action="store_true",
        help="print resolved selections/tests; no SDK/NPU required",
    )
    args = parser.parse_args(argv)
    args.artifacts = args.artifacts.split(",")
    if (
        not args.artifacts
        or len(set(args.artifacts)) != len(args.artifacts)
        or set(args.artifacts).difference(("run", "wheel"))
    ):
        parser.error("--artifacts must be wheel,run, wheel, or run")
    if (
        args.jobs < 1
        or args.npu_device < 0
        or min(args.build_timeout, args.test_timeout) < 0
    ):
        parser.error("jobs must be positive; device/timeouts must be nonnegative")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.list:
        with tempfile.TemporaryDirectory(prefix="sgl-build-matrix-") as temporary:
            inventory, scenarios = matrix(
                Path(temporary), args.arch_version, args.scope
            )
            include_framework_tests(scenarios, args.native_only)
            if args.scenarios:
                names = set(args.scenarios.split(","))
                if names.difference(s["name"] for s in scenarios):
                    raise RuntimeError("Unknown scenario names")
                scenarios = [s for s in scenarios if s["name"] in names]
            print(
                json.dumps(
                    {
                        "arch_version": args.arch_version,
                        "inventory": inventory,
                        "artifacts": args.artifacts,
                        "scenarios": scenarios,
                    },
                    indent=2,
                )
            )
        return 0
    if (
        platform.system() != "Linux"
        or not os.environ.get("ASCEND_HOME_PATH")
        or not Path(os.environ["ASCEND_HOME_PATH"]).is_dir()
    ):
        raise RuntimeError("Run on Linux after sourcing the CANN SDK set_env.sh")
    output = (
        args.results_dir
        or ROOT
        / "output/build-validation"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + args.arch_version)
    ).resolve()
    if output.exists() or any(
        output.is_relative_to(ROOT / "build" / kind)
        for kind in ("cann", "torch", "wheel")
    ):
        raise RuntimeError(
            "Results directory must be new and outside cleaned build directories"
        )
    output.mkdir(parents=True)
    runner = Runner(output)
    runner.context = {
        "arch_version": args.arch_version,
        "sdk": str(Path(os.environ["ASCEND_HOME_PATH"]).resolve()),
        "python": sys.version,
        "started": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    runner.context["git"] = git.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    runner.context["git_status"] = status.stdout.splitlines()
    if (ROOT / "build").is_symlink():
        raise RuntimeError(
            "The build entry does not support a symlinked build directory"
        )
    (ROOT / "build").mkdir(exist_ok=True)
    with (ROOT / "build/.native-validation.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another native validation runner is using this checkout"
            ) from None
        try:
            inventory, scenarios = matrix(
                output / "selection", args.arch_version, args.scope
            )
            include_framework_tests(scenarios, args.native_only)
            if args.scenarios:
                names = set(args.scenarios.split(","))
                if names.difference(s["name"] for s in scenarios):
                    raise RuntimeError("Unknown scenario names; use --list")
                scenarios = [s for s in scenarios if s["name"] in names]
            write_json(
                output / "plan.json",
                {
                    "inventory": inventory,
                    "scenarios": scenarios,
                    "artifacts": args.artifacts,
                },
            )
            for relative in ("version.cfg", "compiler/version.info"):
                metadata = Path(os.environ["ASCEND_HOME_PATH"]) / relative
                if metadata.is_file():
                    runner.context.setdefault("sdk_version_metadata", {})[relative] = (
                        metadata.read_text(errors="replace")
                    )
            print(
                f"Plan: {len(scenarios)} selections x {len(args.artifacts)} artifacts; results: {output}",
                flush=True,
            )
            python = create_environment(output / "environment")
            # Use an isolated process; package smoke checks also use fresh
            # processes after every installation to avoid cached .so/modules.
            preflight = {
                "name": "environment",
                "artifact": "preflight",
                "status": "RUNNING",
                "steps": [],
            }
            runner.cases.append(preflight)
            code = "import torch, torch_npu, setuptools, wheel; print('Torch:',torch.__version__); print('torch_npu:',torch_npu.__version__); print('Python:',__import__('sys').executable)"
            if "wheel" in args.artifacts:
                code += (
                    "; import pytest; assert torch.npu.is_available(), 'NPU unavailable'; torch.npu.set_device("
                    + str(args.npu_device)
                    + "); print('NPU:',torch.npu.get_device_name())"
                )
            result = runner.step(
                preflight, "dependencies", [str(python), "-I", "-c", code], cwd=output
            )
            preflight["status"] = "FAIL" if result["exit_code"] else "PASS"
            if result["exit_code"]:
                runner.context["exit_code"] = 1
                return 1
            for scenario in scenarios:
                for artifact in args.artifacts:
                    case = {
                        "name": scenario["name"],
                        "artifact": artifact,
                        "status": "RUNNING",
                        "ops": scenario["ops"],
                        "selected": scenario["selected"],
                        "steps": [],
                    }
                    runner.cases.append(case)
                    try:
                        execute_case(runner, case, scenario, args, python)
                        if case["status"] == "RUNNING":
                            case["status"] = "PASS"
                    except Exception as error:
                        case["status"] = "FAIL"
                        case["reason"] = str(error)
                        print(f"FAIL: {case['name']}/{artifact}: {error}", flush=True)
                    finally:
                        try:
                            preserve_build_logs(output / case["name"] / artifact)
                        except OSError as error:
                            case["status"] = "FAIL"
                            case["log_copy_error"] = str(error)
                        runner.save()
            runner.context["exit_code"] = int(
                any(case["status"] != "PASS" for case in runner.cases)
            )
            return runner.context["exit_code"]
        except KeyboardInterrupt:
            if runner.cases:
                runner.cases[-1]["status"] = "INTERRUPTED"
            runner.context["exit_code"] = 130
            return 130
        except Exception as error:
            runner.context["error"] = str(error)
            runner.context["exit_code"] = 1
            raise
        finally:
            runner.context["finished"] = datetime.now(timezone.utc).isoformat()
            runner.save()
            print(
                "\nOverall exit code: " + str(runner.context.get("exit_code", 1)),
                flush=True,
            )
            print(f"\nResults: {output}\nSummary: {output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--_check-wheel":
            check_wheel(sys.argv[2])
            sys.exit(0)
        if len(sys.argv) == 3 and sys.argv[1] == "--_check-run":
            check_run(sys.argv[2])
            sys.exit(0)
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
