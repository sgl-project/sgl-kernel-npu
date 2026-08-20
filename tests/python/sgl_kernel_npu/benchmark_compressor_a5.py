# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

"""Ascend950 A5 compressor native-CANN versus migrated-kernel benchmark.

The timed region contains only pre-captured graph replays. Input transfer, state
reset, tiling warmup, graph capture, output handling, and synchronization setup
are deliberately outside the NPU-event interval.
"""

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

try:
    import sgl_kernel_npu  # noqa: F401
    import torch_npu
except ModuleNotFoundError:
    torch_npu = None

from test_compressor_a5 import _call_a5_device_compressor, _make_a5_device_inputs
from test_compressor_a5 import _make_a5_inputs

DECODE = tuple((batch, 1) for batch in (1, 8, 32, 64, 128))
SHORT_PREFILL = tuple(
    (batch, sequence) for batch in (1, 8) for sequence in (16, 128, 512)
)
LONG_PREFILL = tuple((1, sequence) for sequence in (4096, 8192))
SCENARIOS = ((4, 2, 512), (4, 2, 128), (128, 1, 512))
HIDDEN_SIZE = 7168
WARMUP_ITERATIONS = 20
TIMED_ITERATIONS = 100
SAMPLE_COUNT = 3
CASE_REGRESSION_GATE_PCT = 5.0
AGGREGATE_REGRESSION_GATE_PCT = 3.0


@dataclass(frozen=True)
class BenchmarkCase:
    workload: str
    cmp_ratio: int
    coff: int
    head_dim: int
    batch: int
    sequence: int
    start_position: int
    hidden_size: int
    dtype: str
    layout: str
    cache_mode: str
    production_gate: bool


def _scenario_name(cmp_ratio: int, coff: int, head_dim: int) -> str:
    return f"c{cmp_ratio}c{coff}h{head_dim}"


def _production_cases() -> tuple[BenchmarkCase, ...]:
    cases = []
    for workload, matrix in (
        ("decode", DECODE),
        ("short_prefill", SHORT_PREFILL),
        ("long_prefill", LONG_PREFILL),
    ):
        for cmp_ratio, coff, head_dim in SCENARIOS:
            for batch, sequence in matrix:
                cases.append(
                    BenchmarkCase(
                        workload=workload,
                        cmp_ratio=cmp_ratio,
                        coff=coff,
                        head_dim=head_dim,
                        batch=batch,
                        sequence=sequence,
                        start_position=8192 if workload == "decode" else 0,
                        hidden_size=HIDDEN_SIZE,
                        dtype="bfloat16",
                        layout="TH",
                        cache_mode="Cycle",
                        production_gate=True,
                    )
                )
    return tuple(cases)


PRODUCTION_CASES = _production_cases()
BASIC_CASES = (
    BenchmarkCase(
        "basic_fp16",
        4,
        2,
        512,
        1,
        16,
        0,
        1024,
        "float16",
        "TH",
        "Cycle",
        False,
    ),
    BenchmarkCase(
        "basic_bsh",
        4,
        2,
        512,
        1,
        16,
        0,
        1024,
        "bfloat16",
        "BSH",
        "Cycle",
        False,
    ),
    BenchmarkCase(
        "basic_continuous",
        4,
        2,
        512,
        1,
        16,
        0,
        1024,
        "bfloat16",
        "TH",
        "Continuous",
        False,
    ),
)


@dataclass
class GraphRunner:
    graph: object
    state: torch.Tensor
    initial_state: torch.Tensor
    output: torch.Tensor
    start_event: object
    end_event: object

    def reset(self) -> None:
        self.state.copy_(self.initial_state)
        torch.npu.synchronize()


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _cache_mode(name: str) -> int:
    return 2 if name == "Cycle" else 1


def _make_inputs(case: BenchmarkCase) -> dict:
    return _make_a5_inputs(
        start_pos=[case.start_position] * case.batch,
        seq_len=case.sequence,
        coff=case.coff,
        cmp_ratio=case.cmp_ratio,
        head_dim=case.head_dim,
        hidden=case.hidden_size,
        cache_mode=_cache_mode(case.cache_mode),
        layout=case.layout,
        dtype=_dtype(case.dtype),
        batch=case.batch,
        block_size=16,
        noncontiguous_dim0=False,
        seed=20260820,
    )


def _warmup(
    operator: Callable, device_inputs: dict, initial_state: torch.Tensor
) -> None:
    state = initial_state.clone()
    for _ in range(WARMUP_ITERATIONS):
        _call_a5_device_compressor(operator, device_inputs, state)
    torch.npu.synchronize()


def _capture_runner(
    operator: Callable, device_inputs: dict, initial_state: torch.Tensor
) -> GraphRunner:
    state = initial_state.clone()
    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=True):
        output = _call_a5_device_compressor(operator, device_inputs, state)
    torch.npu.synchronize()
    return GraphRunner(
        graph=graph,
        state=state,
        initial_state=initial_state.clone(),
        output=output,
        start_event=torch.npu.Event(enable_timing=True),
        end_event=torch.npu.Event(enable_timing=True),
    )


def _validate_native_migrated(device_inputs: dict, initial_state: torch.Tensor) -> None:
    native_state, migrated_state = initial_state.clone(), initial_state.clone()
    native_output = _call_a5_device_compressor(
        torch.ops.custom.compressor, device_inputs, native_state
    )
    migrated_output = _call_a5_device_compressor(
        torch.ops.npu.compressor, device_inputs, migrated_state
    )
    torch.npu.synchronize()
    if not torch.equal(native_output, migrated_output):
        raise AssertionError("native and migrated compressor outputs differ")
    if not torch.equal(native_state, migrated_state):
        raise AssertionError("native and migrated compressor states differ")


def _time_graph_replay(runner: GraphRunner) -> float:
    # The reset and its synchronization intentionally occur before start.record().
    runner.reset()
    runner.start_event.record()
    for _ in range(TIMED_ITERATIONS):
        runner.graph.replay()
    runner.end_event.record()
    runner.end_event.synchronize()
    return runner.start_event.elapsed_time(runner.end_event) * 1000.0 / TIMED_ITERATIONS


def _percentile_90(samples: list[float]) -> float:
    return statistics.quantiles(samples, n=10, method="inclusive")[8]


def _version_or_unknown(value: object) -> str:
    return str(value) if value is not None else "unknown"


def _cann_version() -> str:
    version = getattr(torch.version, "cann", None) or os.environ.get("ASCEND_VERSION")
    if version is not None:
        return str(version)
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend")
    version_files = (
        Path(ascend_home) / "version.info",
        Path(ascend_home) / "ascend-toolkit" / "latest" / "version.info",
    )
    for version_file in version_files:
        try:
            for line in version_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version="):
                    return line.partition("=")[2]
        except OSError:
            continue
    return "unknown"


def _environment() -> dict[str, str]:
    return {
        "soc_name": torch.npu.get_device_name(0),
        "cann_version": _cann_version(),
        "torch_npu_version": _version_or_unknown(
            getattr(torch_npu, "__version__", None)
        ),
    }


def _profile_replay(runner: GraphRunner, output_dir: Path, label: str) -> None:
    """Write a replay-only trace for manual tiling/copy inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
    handler = torch_npu.profiler.tensorboard_trace_handler(str(output_dir / label))
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]
    runner.reset()
    with torch_npu.profiler.profile(
        activities=activities, schedule=schedule, on_trace_ready=handler
    ) as profiler:
        runner.graph.replay()
        torch.npu.synchronize()
        profiler.step()


def _benchmark_case(case: BenchmarkCase, profile_dir: Path | None) -> dict:
    inputs = _make_inputs(case)
    device_inputs = _make_a5_device_inputs(inputs)
    initial_state = inputs["state_cache"].npu()

    _validate_native_migrated(device_inputs, initial_state)
    _warmup(torch.ops.custom.compressor, device_inputs, initial_state)
    _warmup(torch.ops.npu.compressor, device_inputs, initial_state)

    # Capture three independent state/output-owning graphs per operator before timing.
    runners = {
        "native": tuple(
            _capture_runner(torch.ops.custom.compressor, device_inputs, initial_state)
            for _ in range(SAMPLE_COUNT)
        ),
        "migrated": tuple(
            _capture_runner(torch.ops.npu.compressor, device_inputs, initial_state)
            for _ in range(SAMPLE_COUNT)
        ),
    }
    if profile_dir is not None and (
        case.workload == "decode"
        and case.cmp_ratio == 4
        and case.coff == 2
        and case.batch == 1
    ):
        _profile_replay(runners["native"][0], profile_dir, "decode_c4a_native")
        _profile_replay(runners["migrated"][0], profile_dir, "decode_c4a_migrated")
    if profile_dir is not None and case.workload == "basic_bsh":
        _profile_replay(runners["native"][0], profile_dir, "short_bsh_full_load_native")
        _profile_replay(
            runners["migrated"][0], profile_dir, "short_bsh_full_load_migrated"
        )

    samples = {"native": [], "migrated": []}
    for sample_index in range(SAMPLE_COUNT):
        order = (
            ("native", "migrated")
            if sample_index % 2 == 0
            else (
                "migrated",
                "native",
            )
        )
        for implementation in order:
            samples[implementation].append(
                _time_graph_replay(runners[implementation][sample_index])
            )

    native_median = statistics.median(samples["native"])
    migrated_median = statistics.median(samples["migrated"])
    if native_median <= 0.0:
        raise RuntimeError("native compressor event timing was not positive")
    regression_pct = (migrated_median / native_median - 1.0) * 100.0
    return {
        **_environment(),
        **asdict(case),
        "scenario": _scenario_name(case.cmp_ratio, case.coff, case.head_dim),
        "native_median_us": native_median,
        "native_p90_us": _percentile_90(samples["native"]),
        "migrated_median_us": migrated_median,
        "migrated_p90_us": _percentile_90(samples["migrated"]),
        "regression_pct": regression_pct,
    }


def _require_a5_environment() -> None:
    if torch_npu is None or not hasattr(torch, "npu"):
        raise RuntimeError(
            "torch-npu and the rebuilt sgl_kernel_npu wheel are required"
        )
    if not hasattr(torch.ops.custom, "compressor"):
        raise RuntimeError("torch.ops.custom.compressor is required for native A/B")
    if not hasattr(torch.ops.npu, "compressor"):
        raise RuntimeError("torch.ops.npu.compressor is required for migrated A/B")
    if "ascend950" not in torch.npu.get_device_name(0).replace(" ", "").lower():
        raise RuntimeError("the A5 compressor benchmark requires Ascend950")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="write replay-only profiler traces for representative decode and short-BSH cases",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print the immutable benchmark contract as JSONL without requiring NPU hardware",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cases = PRODUCTION_CASES + BASIC_CASES
    if args.list_cases:
        for case in cases:
            print(json.dumps(asdict(case), sort_keys=True))
        return 0

    _require_a5_environment()
    results = []
    for case in cases:
        result = _benchmark_case(case, args.profile_dir)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    production_regressions = [
        result["regression_pct"] for result in results if result["production_gate"]
    ]
    aggregate_regression = statistics.median(production_regressions)
    summary = {
        "record_type": "aggregate",
        "production_case_count": len(production_regressions),
        "aggregate_production_median_regression_pct": aggregate_regression,
        "aggregate_regression_gate_pct": AGGREGATE_REGRESSION_GATE_PCT,
        "case_regression_gate_pct": CASE_REGRESSION_GATE_PCT,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)

    failures = [
        result
        for result in results
        if result["production_gate"]
        and result["regression_pct"] > CASE_REGRESSION_GATE_PCT
    ]
    if failures or aggregate_regression > AGGREGATE_REGRESSION_GATE_PCT:
        print("A5 compressor performance regression gate failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
