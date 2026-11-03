"""Real benchmark orchestration with a controlled profiler protocol."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.python.utils.performance import benchmark_npu


# Compile once before profiling, then synchronize and step through warmup plus active iterations.
def test_benchmark_separates_initial_execution_and_profile_iterations(
    npu_runtime, tmp_path, monkeypatch
):
    import torch_npu

    events, schedules, profiles, trace_paths = [], [], [], []
    directory = tmp_path / "profile"
    schedule_token, trace_token = object(), object()
    npu_activity = object()

    class Profile:
        def __enter__(self):
            events.append("enter")
            return self

        def step(self):
            events.append("step")

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")
            report = directory / "host_ascend_pt/report/op_summary.csv"
            report.parent.mkdir(parents=True)
            with report.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Task Duration(us)"])
                writer.writerows([[10], [20]])

    def profile(**kwargs):
        profiles.append(kwargs)
        return Profile()

    def schedule(**kwargs):
        schedules.append(kwargs)
        return schedule_token

    def trace_handler(path):
        trace_paths.append(path)
        return trace_token

    profiler_api = SimpleNamespace(
        profile=profile,
        schedule=schedule,
        tensorboard_trace_handler=trace_handler,
        ProfilerActivity=SimpleNamespace(NPU=npu_activity),
        ProfilerLevel=SimpleNamespace(Level2=object()),
        AiCMetrics=SimpleNamespace(PipeUtilization=object()),
        _ExperimentalConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(torch_npu, "profiler", profiler_api, raising=False)
    monkeypatch.setattr(
        npu_runtime.torch.npu,
        "synchronize",
        lambda: events.append("sync"),
        raising=False,
    )
    average = benchmark_npu(
        lambda: events.append("execute"), directory, warmup=2, active=3
    )

    assert events == ["execute", "sync", "enter"] + [
        "sync",
        "execute",
        "sync",
        "step",
    ] * 5 + ["exit"]
    assert schedules == [
        {"wait": 0, "warmup": 2, "active": 3, "repeat": 1, "skip_first": 0}
    ]
    assert trace_paths == [str(directory)]
    assert len(profiles) == 1
    assert profiles[0]["activities"] == [npu_activity]
    assert profiles[0]["schedule"] is schedule_token
    assert profiles[0]["on_trace_ready"] is trace_token
    assert average == 10
    assert (directory / "report/op_summary.csv").is_file()
    assert not (directory / "host_ascend_pt").exists()


# Stop on executor failure, exit an entered profiler, and preserve artifacts without parsing results.
@pytest.mark.parametrize("failure_call", [1, 2, 3], ids=["initial", "warmup", "active"])
def test_benchmark_executor_failure_stops_and_exits_profiler(
    npu_runtime, tmp_path, monkeypatch, failure_call
):
    import torch_npu

    from tests.python.utils import performance

    directory = tmp_path / "failed-profile"
    error = RuntimeError("executor failed")
    events, exits = [], []

    class Profile:
        def __enter__(self):
            events.append("enter")
            (directory / "partial-trace.txt").write_text("partial trace")
            return self

        def step(self):
            events.append("step")

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")
            exits.append((exc_type, exc_value, traceback))

    profile = Mock(return_value=Profile())
    profiler_api = SimpleNamespace(
        profile=profile,
        schedule=Mock(),
        tensorboard_trace_handler=Mock(),
        ProfilerActivity=SimpleNamespace(NPU=object()),
        ProfilerLevel=SimpleNamespace(Level2=object()),
        AiCMetrics=SimpleNamespace(PipeUtilization=object()),
        _ExperimentalConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(torch_npu, "profiler", profiler_api, raising=False)
    monkeypatch.setattr(
        npu_runtime.torch.npu, "synchronize", lambda: events.append("sync")
    )
    flatten = Mock(wraps=performance.flatten_profile_directory)
    read_time = Mock(wraps=performance.read_device_time_us)
    monkeypatch.setattr(performance, "flatten_profile_directory", flatten)
    monkeypatch.setattr(performance, "read_device_time_us", read_time)

    def execute():
        events.append("execute")
        if events.count("execute") == failure_call:
            raise error

    with pytest.raises(RuntimeError) as caught:
        benchmark_npu(execute, directory, warmup=1, active=2)

    assert caught.value is error
    flatten.assert_not_called()
    read_time.assert_not_called()
    assert directory.is_dir()
    if failure_call == 1:
        assert events == ["execute"]
        profile.assert_not_called()
        profiler_api.schedule.assert_not_called()
        assert exits == []
        assert list(directory.iterdir()) == []
    else:
        completed_iterations = failure_call - 2
        assert events == (
            ["execute", "sync", "enter"]
            + ["sync", "execute", "sync", "step"] * completed_iterations
            + ["sync", "execute", "exit"]
        )
        profile.assert_called_once()
        assert len(exits) == 1
        exc_type, exc_value, traceback = exits[0]
        assert exc_type is RuntimeError
        assert exc_value is error
        assert traceback is not None
        assert (directory / "partial-trace.txt").read_text() == "partial trace"


# Refuse an existing profiling directory before executing the operator or overwriting artifacts.
def test_existing_profile_directory_is_not_overwritten(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("existing profile")
    calls = []
    with pytest.raises(FileExistsError):
        benchmark_npu(lambda: calls.append("execute"), directory, warmup=1, active=1)
    assert calls == []
    assert marker.read_text() == "existing profile"
    assert list(directory.iterdir()) == [marker]


# Validate both sampling counts before directory creation or operator execution.
@pytest.mark.parametrize(
    "warmup, active, field, error",
    [
        (0, 5, "warmup", ValueError),
        (5, 0, "active", ValueError),
        (5, 1.5, "active", TypeError),
    ],
    ids=["zero-warmup", "zero-active", "fractional-active"],
)
def test_benchmark_rejects_invalid_sampling_before_execution(
    tmp_path, warmup, active, field, error
):
    directory = tmp_path / "must-not-exist"
    calls = []
    with pytest.raises(error, match=f"{field} must be a positive integer"):
        benchmark_npu(
            lambda: calls.append("execute"), directory, warmup=warmup, active=active
        )
    assert calls == []
    assert not directory.exists()
