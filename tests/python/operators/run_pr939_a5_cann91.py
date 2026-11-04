#!/usr/bin/env python3
"""PR #939 A5/CANN 9.1 accuracy runner with durable per-case summaries.

Usage: python3 test_cann91_a5.py /path/to/sgl-kernel-npu --device 0
Native success must be established on the server; collection is not accuracy.
"""
import argparse
import csv
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PLUGIN = r"""
import json
import os
from pathlib import Path

def emit(kind, **kwargs):
    with Path(os.environ['PR939_EVENTS']).open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(kind=kind, **kwargs), ensure_ascii=False) + '\n')

def pytest_collection_finish(session):
    emit('collected', nodes=[item.nodeid for item in session.items])

def pytest_runtest_logstart(nodeid, location):
    emit('start', node=nodeid)

def pytest_runtest_logreport(report):
    emit('report', node=report.nodeid, when=report.when,
         outcome=report.outcome, duration=report.duration,
         reason=str(report.longrepr)[:2000] if report.failed or report.skipped else '')

def pytest_collectreport(report):
    if report.failed or report.skipped:
        emit('collection', node=report.nodeid, outcome=report.outcome,
             reason=str(report.longrepr)[:2000])
"""

PREFLIGHT = r"""
import importlib
import json
import sys
import torch
import torch_npu
import sgl_kernel_npu
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware
try:
    from sgl_kernel_npu.implementation.torch_native.kvcacheio._quantization import _fp8_to_float
except ImportError as exc:
    raise RuntimeError('Installed wheel lacks the CANN 9.1 FP8 reference fix; install the updated PR wheel') from exc
from sgl_kernel_npu.operators.utils._transformer_compat import detect_cann_version, cann_backend, optional_op
if not torch.npu.is_available():
    raise RuntimeError('NPU unavailable')
torch.npu.set_device(int(sys.argv[1]))
if get_hardware().generation != Generation.ARCH35:
    raise RuntimeError('Requires A5 / ARCH35')
version = detect_cann_version()
if not version.startswith('9.1.') or cann_backend() != 'legacy':
    raise RuntimeError('Requires initialized CANN 9.1.x and legacy backend')
result = dict(python=sys.executable, torch=torch.__version__,
              torch_npu=torch_npu.__file__, package=sgl_kernel_npu.__file__,
              cann=version, device=int(sys.argv[1]),
              device_name=torch.npu.get_device_name(int(sys.argv[1])), bindings={})
for namespace, name in (
    ('custom', 'npu_hc_pre'), ('custom', 'npu_hc_post'),
    ('custom', 'inplace_partial_rotary_mul'),
    ('custom', 'npu_kv_quant_sparse_attn_sharedkv_metadata'),
    ('custom', 'npu_kv_quant_sparse_attn_sharedkv'),
    ('custom', 'npu_quant_lightning_indexer_metadata'),
    ('custom', 'npu_quant_lightning_indexer'),
    ('custom', 'indexer_compress_epilog'), ('custom', 'compressor'),
    ('custom', 'npu_moe_gating_top_k'), ('npu', 'kv_compress_epilog'),
):
    key = namespace + '.' + name
    try:
        op = optional_op(namespace, name)
        result['bindings'][key] = dict(available=op is not None,
             schema=str(op.default._schema) if op is not None else '')
    except Exception as exc:
        result['bindings'][key] = dict(available=False, error=repr(exc))
print(json.dumps(result, ensure_ascii=False, indent=2))
"""


def plan(repo):
    base = "tests/python/operators/"
    tasks = []
    split = (repo / (base + "norm/test_mhc_pre.py")).is_file()
    tasks.extend(
        [
            (
                "mhc_pre",
                base
                + (
                    "norm/test_mhc_pre.py"
                    if split
                    else "norm/test_mhc.py::test_mhc_pre"
                ),
                [],
            ),
            (
                "mhc_post",
                base
                + (
                    "norm/test_mhc_post.py"
                    if split
                    else "norm/test_mhc.py::test_mhc_post"
                ),
                [],
            ),
            ("rope", base + "rope/test_inplace_partial_rotary_mul.py", []),
            ("kv_writer", base + "kvcacheio/test_kv_compress_epilog.py", []),
            ("indexer_writer", base + "kvcacheio/test_indexer_quant_cache.py", []),
            (
                "quant_attention",
                base + "attention/test_mixed_quant_sparse_flash_mla.py",
                [],
            ),
        ]
    )
    meta = base + "attention/test_mixed_quant_sparse_flash_mla_metadata.py"
    if (repo / meta).is_file():
        tasks.append(("quant_attention_metadata", meta, []))
    tasks.extend(
        [
            ("indexer", base + "attention/test_quant_lightning_indexer.py", []),
            (
                "indexer_metadata",
                base + "attention/test_quant_lightning_indexer_metadata.py",
                [],
            ),
            # BSH is not validated by this migration; retain TH, both cache modes,
            # all three compression shapes, batching, and streaming state coverage.
            ("compressor", base + "attention/test_compressor.py", ["-k", "not BSH"]),
            ("moe", base + "moe/test_moe_gating_top_k.py", []),
        ]
    )
    return tasks


def terminate(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def execute(command, log, cwd, env, timeout):
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            terminate(proc)
            return 124, True
        except KeyboardInterrupt:
            terminate(proc)
            return 130, False


def read_events(path, group, code, timed_out):
    rows = {}
    active = None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # A killed process can leave a truncated last line.
            kind = item["kind"]
            if kind == "collected":
                for node in item["nodes"]:
                    rows.setdefault(
                        node, dict(group=group, test=node, status="NOT_RUN", reason="")
                    )
            elif kind == "start":
                active = item["node"]
            elif kind in ("collection", "report"):
                node = item["node"]
                row = rows.setdefault(
                    node, dict(group=group, test=node, status="NOT_RUN", reason="")
                )
                outcome = item["outcome"]
                if kind == "collection":
                    row.update(
                        status="SKIP" if outcome == "skipped" else "ERROR",
                        reason=item["reason"],
                    )
                elif outcome == "failed":
                    row.update(
                        status="FAIL" if item["when"] == "call" else "ERROR",
                        reason=item["reason"],
                    )
                elif outcome == "skipped" and row["status"] not in ("FAIL", "ERROR"):
                    row.update(status="SKIP", reason=item["reason"])
                elif item["when"] == "call" and row["status"] == "NOT_RUN":
                    row["status"] = "PASS"
                if kind == "report" and item["when"] == "teardown":
                    active = None
    if timed_out or code not in (0, 1, 5):
        node = active or group + "::<process>"
        row = rows.setdefault(
            node, dict(group=group, test=node, status="NOT_RUN", reason="")
        )
        row.update(
            status="TIMEOUT" if timed_out else "ERROR",
            reason=f"process exit={code}; see group log",
        )
    elif not rows or (
        code != 0 and all(r["status"] in ("PASS", "SKIP") for r in rows.values())
    ):
        rows[group + "::<process>"] = dict(
            group=group,
            test=group + "::<process>",
            status="ERROR",
            reason=f"pytest exit={code}; no complete success report; see group log",
        )
    return list(rows.values())


def save_summary(directory, rows):
    (directory / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (directory / "summary.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["group", "test", "status", "reason"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = ["STATUS     TEST", "-" * 100]
    for row in rows:
        lines.append(f"{row['status']:<10} {row['test']}")
    counts = Counter(row["status"] for row in rows)
    lines.extend(
        [
            "",
            "TOTAL: "
            + "  ".join(
                f"{key}={counts[key]}"
                for key in (
                    "PASS",
                    "FAIL",
                    "ERROR",
                    "SKIP",
                    "TIMEOUT",
                    "NOT_RUN",
                    "EXCLUDED",
                )
            ),
            "EXCLUDED/SKIP do not count as passed accuracy checks.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (directory / "summary.txt").write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default=".")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Seconds per test file; timeout is recorded as failure",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device < 0 or args.timeout < 1:
        parser.error("device must be nonnegative and timeout positive")
    repo = Path(args.repo).resolve()
    if not (repo / "tests/python/conftest.py").is_file():
        parser.error("repo must be the PR #939 sgl-kernel-npu checkout root")
    directory = (
        args.output
        or repo
        / "test-results"
        / (
            "a5-cann91-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + str(os.getpid())
        )
    ).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    plugin_dir = directory / "runner-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "pr939_a5_events.py").write_text(PLUGIN, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(plugin_dir) + os.pathsep + env.get("PYTHONPATH", "")
    rows = [
        dict(
            group="excluded",
            test="sparse_flash_mla",
            status="EXCLUDED",
            reason="PR #47438 routes A5 to quantized shared-KV; this installed 9.1 recipe lacks the BF16 shared-KV symbol",
        ),
        dict(
            group="excluded",
            test="sparse_flash_mla_metadata",
            status="EXCLUDED",
            reason="Same A5 dispatch constraint; not an A5 legacy required path",
        ),
        dict(
            group="excluded",
            test="compressor::<BSH>",
            status="EXCLUDED",
            reason="This suite covers TH only; BSH is not established by the current migration",
        ),
    ]
    print(f"Results: {directory}", flush=True)
    print(
        "Checking A5/CANN 9.1 environment; binding presence is not kernel execution validation.",
        flush=True,
    )
    status, timed_out = execute(
        [sys.executable, "-c", PREFLIGHT, str(args.device)],
        directory / "environment.log",
        repo,
        env,
        args.timeout,
    )
    print(
        (directory / "environment.log").read_text(encoding="utf-8", errors="replace"),
        flush=True,
    )
    if status != 0:
        rows.append(
            dict(
                group="environment",
                test="preflight",
                status="TIMEOUT" if timed_out else "ERROR",
                reason=f"exit={status}; see environment.log",
            )
        )
        print(save_summary(directory, rows))
        return 2
    for group, node, extra in plan(repo):
        if not (repo / node.split("::")[0]).is_file():
            rows.append(
                dict(
                    group=group,
                    test=node,
                    status="ERROR",
                    reason="Missing test source; update checkout",
                )
            )
            save_summary(directory, rows)
            continue
        event = directory / (group + ".jsonl")
        case_env = dict(env, PR939_EVENTS=str(event))
        command = [
            sys.executable,
            "-u",
            "-m",
            "pytest",
            node,
            *extra,
            "-p",
            "pr939_a5_events",
            "--impl=auto",
            "--impl-ref=torch_native",
            f"--npu-device={args.device}",
            "--skip-acc-check=false",
            "--enable-perf=false",
            "-o",
            "faulthandler_timeout=60",
            "-v",
            "-s",
            "-ra",
            "--tb=short",
            f'--junitxml={directory / (group + ".xml")}',
        ]
        print(f"RUN {group}: {node} (timeout {args.timeout}s)", flush=True)
        code, timed_out = execute(
            command, directory / (group + ".log"), repo, case_env, args.timeout
        )
        group_rows = read_events(event, group, code, timed_out)
        rows.extend(group_rows)
        print(
            "  "
            + " ".join(
                f"{key}={value}"
                for key, value in Counter(r["status"] for r in group_rows).items()
            ),
            flush=True,
        )
        save_summary(directory, rows)
        if code == 130:
            break
    print("\n" + save_summary(directory, rows), flush=True)
    print(f"Full logs, JUnit XML, summary.txt/.tsv/.json: {directory}", flush=True)
    failures = {"FAIL", "ERROR", "TIMEOUT", "NOT_RUN"}
    return int(
        any(r["status"] in failures for r in rows)
        or not any(r["status"] == "PASS" for r in rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
