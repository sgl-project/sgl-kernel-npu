"""NPU profiling, optional CSV results and automatic pytest reporting."""

import csv
import json
import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from pathlib import Path
from uuid import uuid4

_REPORT_CONTEXT = ContextVar("performance_report_context", default=None)
_ENABLE_PERF = ContextVar("enable_performance", default=None)
_OUTPUT_PATHS = ContextVar("performance_output_paths", default=(None, None))
_PROFILE_SEQUENCE = count(1)


@contextmanager
def performance_context(
    test_case,
    parameters,
    record_property,
    *,
    enable_perf=None,
    prof_dir=None,
    csv_path=None,
):
    token = _REPORT_CONTEXT.set((test_case, dict(parameters), record_property))
    perf_token = _ENABLE_PERF.set(enable_perf)
    paths_token = _OUTPUT_PATHS.set((prof_dir, csv_path))
    try:
        yield
    finally:
        _OUTPUT_PATHS.reset(paths_token)
        _ENABLE_PERF.reset(perf_token)
        _REPORT_CONTEXT.reset(token)


def read_device_time_us(directory, active):
    """Sum device task durations per call, not wall time or critical-path latency."""
    if active <= 0:
        raise ValueError("active must be positive")
    directory = Path(directory)
    files = sorted(directory.rglob("op_summary*.csv"))
    column = "Task Duration(us)"
    if not files:
        files = sorted(directory.rglob("op_statistic.csv"))
        column = "Total Time(us)"
    if not files:
        raise RuntimeError(f"No NPU profiler CSV found under {directory}")
    total = 0.0
    count = 0
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if column not in (reader.fieldnames or []):
                raise RuntimeError(f"Missing {column!r} in {path}")
            for row in reader:
                duration = float(row[column])
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError(f"Invalid duration in {path}: {duration}")
                total += duration
                count += 1
    if not count or total <= 0:
        raise RuntimeError("Profiler returned no positive device task duration")
    return total / active


def flatten_profile_directory(directory):
    """Remove a single profiler-generated host directory, retaining all contents."""
    directory = Path(directory)
    runs = [
        path
        for path in directory.iterdir()
        if path.is_dir() and path.name.endswith("_ascend_pt")
    ]
    if len(runs) != 1:
        return
    run = runs[0]
    children = list(run.iterdir())
    if any((directory / child.name).exists() for child in children):
        return
    for child in children:
        child.rename(directory / child.name)
    run.rmdir()


def profile_case_name(parameters, impl):
    """Order profiles by creation, followed by compact values and a unique ID."""

    def safe(value):
        if isinstance(value, (tuple, list)):
            return "x".join(safe(item) for item in value)
        return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value).removeprefix("torch.")).strip(
            "-"
        )

    params = (
        "-".join(
            safe(value)
            for key, value in sorted(parameters.items())
            if key != "impl_pair"
        )
        or "default"
    )
    # Bound the component length for filesystems with a 255-byte name limit.
    return f"{next(_PROFILE_SEQUENCE):06d}-{params[:160]}-{safe(impl)[:40]}-{uuid4().hex[:8]}"


def benchmark_npu(executor, directory, warmup=5, active=5):
    """Compile/warm up before collecting only this executor's NPU tasks."""
    import torch
    import torch_npu

    if warmup < 1 or active < 1:
        raise ValueError("warmup and active must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    executor()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=warmup,
            active=active,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False,
        ),
    ) as profiler:
        for _ in range(warmup + active):
            torch.npu.synchronize()
            executor()
            torch.npu.synchronize()
            profiler.step()
    flatten_profile_directory(directory)
    return read_device_time_us(directory, active)


def write_result(path, **result):
    """Append serial test results, checking the existing CSV schema."""
    if "average_device_task_us" in result:
        result["average_device_task_us"] = f"{result['average_device_task_us']:.3f}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as stream:
        stream.seek(0)
        header = next(csv.reader(stream), None)
        if header is not None and header != list(result):
            raise ValueError(f"CSV columns do not match existing file: {path}")
        stream.seek(0, 2)
        writer = csv.DictWriter(stream, fieldnames=list(result))
        if header is None:
            writer.writeheader()
        writer.writerow(result)


def check_performance(
    executor,
    csv_path=None,
    *,
    enable_perf=None,
    prof_dir=None,
    warmup=5,
    active=5,
):
    """Profile an executor returned by check_accuracy; optionally append CSV.

    Profiler files remain in <prof_dir>/prof_<function>/<params_impl_run>/, even on
    failure. prof_dir defaults to cwd/profilings; relative paths use cwd.
    The optional summary CSV is written only when csv_path is provided.
    An explicit --enable-perf=true/false overrides the code's enable_perf.
    Explicit --prof-dir and --csv-path override the corresponding code paths.
    Results include accuracy_checked: True only when the executor records a
    successful accuracy check; skipped or missing status is recorded as False.
    If neither is specified, performance is disabled. Disabled calls return
    None without profiling or writing files.
    """
    cli_enable_perf = _ENABLE_PERF.get()
    if cli_enable_perf is not None:
        enable_perf = cli_enable_perf
    elif enable_perf is None:
        enable_perf = False
    if not isinstance(enable_perf, bool):
        raise TypeError("enable_perf must be bool or None")
    if not enable_perf:
        return None

    cli_prof_dir, cli_csv_path = _OUTPUT_PATHS.get()
    if cli_prof_dir is not None:
        prof_dir = cli_prof_dir
    if cli_csv_path is not None:
        csv_path = cli_csv_path

    import torch
    from sgl_kernel_npu.operators.utils.registry import get_op_name

    op_name = get_op_name(executor.func)
    context = _REPORT_CONTEXT.get()
    test_case, parameters, record = context if context else ("", {}, None)
    function_name = re.sub(r"[^A-Za-z0-9_.-]", "_", executor.func.__name__)
    run_id = profile_case_name(parameters, executor.impl)
    profile_root = (
        Path.cwd() / "profilings"
        if prof_dir is None
        else Path(prof_dir).expanduser().resolve()
    )
    profiling_dir = profile_root / f"prof_{function_name}" / run_id
    print(f"Profiling output: {profiling_dir}")
    average_us = benchmark_npu(
        executor,
        profiling_dir,
        warmup=warmup,
        active=active,
    )
    result = {
        "test_case": test_case,
        "operator": op_name,
        "impl": executor.impl,
        "impl_ref": executor.impl_ref,
        "accuracy_checked": getattr(executor, "accuracy_checked", False) is True,
        "parameters": json.dumps(parameters, default=str, sort_keys=True),
        "profiling_dir": str(profiling_dir),
        "average_device_task_us": average_us,
        "device": str(torch.device("npu", torch.npu.current_device())),
        "device_name": torch.npu.get_device_name(),
        "warmup": warmup,
        "active": active,
    }
    if csv_path is not None:
        write_result(csv_path, **result)
    if record is not None:
        for key, value in result.items():
            record(key, value)
    print(
        f"{test_case or op_name} [{executor.impl} vs {executor.impl_ref}]: "
        f"{average_us:.3f} us, accuracy_checked={result['accuracy_checked']}"
    )
    return result
