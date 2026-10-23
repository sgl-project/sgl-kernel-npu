"""Real NPU profiler integration, selected separately with -m npu_integration."""

import csv
import math
from pathlib import Path

import pytest

from tests.python.utils.performance import check_performance, performance_context


# Exercise real NPU accuracy, profiling, CSV parsing and reporting without measurement stubs.
@pytest.mark.npu_integration
def test_real_npu_accuracy_and_performance_pipeline(accuracy_case, tmp_path):
    case = accuracy_case
    case.add_impl("dut")
    case.add_impl("ref")
    executor = case.check_accuracy(
        case.operator, args=(case.x,), impl="dut", impl_ref="ref"
    )
    records = []
    csv_path = tmp_path / "results.csv"
    with performance_context(
        "framework::real_npu",
        {"shape": (2,)},
        lambda key, value: records.append((key, value)),
    ):
        result = check_performance(
            executor,
            csv_path,
            enable_perf=True,
            prof_dir=tmp_path / "profiles",
            warmup=1,
            active=2,
        )
    assert math.isfinite(result["average_device_task_us"])
    assert result["average_device_task_us"] > 0
    assert result["accuracy_checked"] is True
    assert result["operator"] == case.op_name
    assert result["impl"] == "dut"
    assert dict(records) == result
    directory = Path(result["profiling_dir"])
    assert directory.is_dir()
    assert list(directory.rglob("op_summary*.csv")) or list(
        directory.rglob("op_statistic.csv")
    )
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["accuracy_checked"] == "True"
    assert rows[0]["profiling_dir"] == str(directory)
