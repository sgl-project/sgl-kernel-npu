"""Performance control and real reporting with a substituted measurement boundary."""

import csv
import json
from contextlib import nullcontext
from pathlib import Path

import pytest


# Keep performance disabled by default even when output paths are supplied.
def test_disabled_performance_has_no_side_effects(performance_case, tmp_path):
    case = performance_case
    result = case.performance.check_performance(
        case.executor, csv_path=tmp_path / "results.csv", prof_dir=tmp_path / "profiles"
    )
    assert result is None
    assert case.measurements == []
    assert case.calls == []
    assert list(tmp_path.iterdir()) == []


# Apply the context switch before deciding whether to measure the executor.
@pytest.mark.parametrize(
    "context_enabled, code_enabled", [(False, True), (True, False)]
)
def test_performance_context_overrides_code_switch(
    performance_case, context_enabled, code_enabled
):
    case = performance_case
    with case.performance.performance_context(
        "case", {}, None, enable_perf=context_enabled
    ):
        result = case.performance.check_performance(
            case.executor, enable_perf=code_enabled
        )
    assert (result is not None) is context_enabled
    assert len(case.measurements) == int(context_enabled)
    assert case.calls == []


# Reject integer flags instead of treating them as booleans.
def test_invalid_effective_performance_flag_is_rejected(performance_case):
    case = performance_case
    with pytest.raises(TypeError, match="enable_perf must be bool or None"):
        case.performance.check_performance(case.executor, enable_perf=1)
    assert case.measurements == []
    assert case.calls == []


# Preserve measurement metadata and record accuracy only when explicitly marked True.
@pytest.mark.parametrize("accuracy_state", [True, False, "missing"])
def test_enabled_performance_reports_measurement_and_metadata(
    performance_case, npu_runtime, tmp_path, accuracy_state
):
    case = performance_case
    if accuracy_state == "missing":
        del case.executor.accuracy_checked
    else:
        case.executor.accuracy_checked = accuracy_state
    records = []
    parameters = {"shape": (2,), "dtype": npu_runtime.torch.float16}
    with case.performance.performance_context(
        "framework::report", parameters, lambda key, value: records.append((key, value))
    ):
        result = case.performance.check_performance(
            case.executor, enable_perf=True, warmup=2, active=3
        )

    directory = Path(result["profiling_dir"])
    assert case.measurements == [(case.executor, directory, 2, 3)]
    assert directory.parent.parent == tmp_path / "profilings"
    assert directory.is_dir()
    assert result["average_device_task_us"] == 12.5
    assert result["operator"] == case.op_name
    assert result["impl"] == "dut"
    assert result["impl_ref"] == "ref"
    assert result["accuracy_checked"] is (accuracy_state is True)
    assert result["test_case"] == "framework::report"
    assert json.loads(result["parameters"]) == {"shape": [2], "dtype": "torch.float16"}
    assert result["device"] == f"npu:{npu_runtime.torch.npu.current_device()}"
    assert result["device_name"] == npu_runtime.soc_name
    assert (result["warmup"], result["active"]) == (2, 3)
    assert dict(records) == result
    assert len(records) == len(result)
    assert not list(tmp_path.rglob("*.csv"))
    assert case.calls == []


# Write results to code paths unless the context supplies higher-priority paths.
@pytest.mark.parametrize("override", [False, True], ids=["code-paths", "context-paths"])
def test_performance_paths_follow_context_precedence(
    performance_case, tmp_path, override
):
    case = performance_case
    code_root, context_root = tmp_path / "code_profiles", tmp_path / "context_profiles"
    code_csv, context_csv = tmp_path / "code.csv", tmp_path / "context.csv"
    with case.performance.performance_context(
        "paths",
        {},
        None,
        prof_dir=context_root if override else None,
        csv_path=context_csv if override else None,
    ):
        result = case.performance.check_performance(
            case.executor, enable_perf=True, prof_dir=code_root, csv_path=code_csv
        )
    expected_root = context_root if override else code_root
    expected_csv = context_csv if override else code_csv
    unused_root = code_root if override else context_root
    unused_csv = code_csv if override else context_csv
    assert Path(result["profiling_dir"]).parent.parent == expected_root
    with expected_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["operator"] == case.op_name
    assert rows[0]["average_device_task_us"] == "12.500"
    assert not unused_root.exists()
    assert not unused_csv.exists()


# Preserve failed profiling artifacts without appending success records or CSV rows.
def test_benchmark_failure_does_not_publish_success(
    performance_case, tmp_path, monkeypatch
):
    case = performance_case
    error = RuntimeError("measurement failed")
    artifacts = []

    def fail(executor, directory, **kwargs):
        directory.mkdir(parents=True)
        artifact = directory / "failure.txt"
        artifact.write_text("failure details")
        artifacts.append(artifact)
        raise error

    monkeypatch.setattr(case.performance, "benchmark_npu", fail)
    csv_path = tmp_path / "existing.csv"
    csv_path.write_text("operator,average_device_task_us\nprevious,1.000\n")
    before = csv_path.read_bytes()
    records = []
    with case.performance.performance_context(
        "failure", {}, lambda key, value: records.append((key, value))
    ):
        with pytest.raises(RuntimeError) as caught:
            case.performance.check_performance(
                case.executor, csv_path, enable_perf=True
            )
    assert caught.value is error
    assert csv_path.read_bytes() == before
    assert records == []
    assert len(artifacts) == 1
    assert artifacts[0].read_text() == "failure details"
    assert case.calls == []


# Restore nested switches, paths and report callbacks after normal or exceptional exits.
@pytest.mark.parametrize("raise_inside", [False, True], ids=["normal", "exception"])
def test_performance_context_is_restored(performance_case, tmp_path, raise_inside):
    case = performance_case
    outer_records, inner_records = [], []
    error = RuntimeError("leave inner performance context")
    with case.performance.performance_context(
        "outer",
        {"scope": "outer"},
        lambda key, value: outer_records.append((key, value)),
        enable_perf=True,
        prof_dir=tmp_path / "outer",
        csv_path=tmp_path / "outer.csv",
    ):
        with pytest.raises(RuntimeError) if raise_inside else nullcontext() as caught:
            with case.performance.performance_context(
                "inner",
                {"scope": "inner"},
                lambda key, value: inner_records.append((key, value)),
                enable_perf=False,
                prof_dir=tmp_path / "inner",
                csv_path=tmp_path / "inner.csv",
            ):
                assert (
                    case.performance.check_performance(case.executor, enable_perf=True)
                    is None
                )
                if raise_inside:
                    raise error
        if raise_inside:
            assert caught.value is error
        outer = case.performance.check_performance(case.executor, enable_perf=False)
    assert case.performance.check_performance(case.executor) is None
    assert outer["test_case"] == "outer"
    assert json.loads(outer["parameters"]) == {"scope": "outer"}
    assert Path(outer["profiling_dir"]).parent.parent == tmp_path / "outer"
    assert inner_records == []
    assert dict(outer_records) == outer
    assert len(outer_records) == len(outer)
    assert not (tmp_path / "inner.csv").exists()
    assert not (tmp_path / "inner").exists()
    assert (tmp_path / "outer.csv").is_file()
    assert len(case.measurements) == 1
