"""Parse real profiler CSVs and preserve real output files."""

import csv
import json
import re

import pytest

from tests.python.utils.performance import (
    flatten_profile_directory,
    profile_case_name,
    read_device_time_us,
    write_result,
)


def write_csv(path, column, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([column])
        writer.writerows([value] for value in values)


# Sum all summary files and ignore statistics when summaries already exist.
def test_summary_files_are_summed_without_statistics_double_counting(tmp_path):
    write_csv(tmp_path / "device0/op_summary_1.csv", "Task Duration(us)", [10])
    write_csv(tmp_path / "device1/op_summary_2.csv", "Task Duration(us)", [20])
    write_csv(tmp_path / "op_statistic.csv", "Total Time(us)", [999])
    assert read_device_time_us(tmp_path, active=2) == 15


# Fall back to the statistics duration column only when summaries are absent.
def test_statistics_are_used_when_summary_is_absent(tmp_path):
    write_csv(tmp_path / "report/op_statistic.csv", "Total Time(us)", [30])
    assert read_device_time_us(tmp_path, active=2) == 15


# Reject missing, malformed, empty and zero-total reports instead of reporting success.
@pytest.mark.parametrize("kind", ["missing", "column", "empty", "zero"])
def test_missing_or_unusable_reports_are_rejected(tmp_path, kind):
    if kind == "column":
        write_csv(tmp_path / "op_summary.csv", "Wrong Column", [10])
        write_csv(tmp_path / "op_statistic.csv", "Total Time(us)", [30])
        message = "Missing"
    elif kind in ("empty", "zero"):
        write_csv(
            tmp_path / "op_summary.csv",
            "Task Duration(us)",
            [] if kind == "empty" else [0],
        )
        message = "no positive device task duration"
    else:
        message = "No NPU profiler CSV"
    with pytest.raises(RuntimeError, match=message):
        read_device_time_us(tmp_path, active=2)


# Reject nonnumeric, negative and nonfinite task durations.
@pytest.mark.parametrize("duration", ["invalid", "-1", "NaN", "Inf"])
def test_invalid_task_durations_are_rejected(tmp_path, duration):
    write_csv(tmp_path / "op_summary.csv", "Task Duration(us)", [duration])
    with pytest.raises(ValueError):
        read_device_time_us(tmp_path, active=2)


# Create and append a CSV with one header, intact quoted fields and rounded durations.
def test_result_csv_can_be_created_and_appended(tmp_path):
    path = tmp_path / "nested/results.csv"
    parameters = json.dumps({"label": 'shape, "edge"', "shape": [1, 7]})
    write_result(
        path,
        operator="test.first",
        parameters=parameters,
        average_device_task_us=12.34567,
    )
    write_result(
        path, operator="test.second", parameters=parameters, average_device_task_us=2
    )
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "operator": "test.first",
            "parameters": parameters,
            "average_device_task_us": "12.346",
        },
        {
            "operator": "test.second",
            "parameters": parameters,
            "average_device_task_us": "2.000",
        },
    ]
    with path.open(newline="") as stream:
        assert len(list(csv.reader(stream))) == 3


# Leave existing results unchanged when a new result has an incompatible schema.
def test_incompatible_csv_schema_preserves_existing_results(tmp_path):
    path = tmp_path / "results.csv"
    write_result(path, operator="existing", average_device_task_us=10)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="CSV columns do not match"):
        write_result(
            path, operator="new", extra="unexpected", average_device_task_us=20
        )
    assert path.read_bytes() == before


# Flatten one host directory completely or preserve all contents if a conflict exists.
@pytest.mark.parametrize(
    "conflict", [False, True], ids=["flatten", "preserve-conflict"]
)
def test_profile_directory_flattening_preserves_files(tmp_path, conflict):
    run = tmp_path / "host_ascend_pt"
    run.mkdir()
    (run / "trace.json").write_text("trace")
    write_csv(run / "report/op_summary.csv", "Task Duration(us)", [10])
    if conflict:
        (tmp_path / "trace.json").write_text("existing")
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    flatten_profile_directory(tmp_path)
    after = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    if conflict:
        assert after == before
        assert run.is_dir()
    else:
        assert not run.exists()
        assert after == {
            p.relative_to("host_ascend_pt"): value for p, value in before.items()
        }


# Generate unique filesystem-safe profile names without depending on fixed UUIDs or counters.
def test_profile_case_names_are_safe_bounded_and_unique():
    parameters = {
        "shape": (7, 1025),
        "dtype": "torch.float16",
        "label": "../边界\\case/" + "x" * 300,
    }
    first = profile_case_name(parameters, "backend/" + "y" * 100)
    second = profile_case_name(parameters, "backend/" + "y" * 100)
    assert first != second
    for name in (first, second):
        assert re.fullmatch(r"[A-Za-z0-9.-]+", name)
        assert "/" not in name and "\\" not in name
        assert len(name.encode("utf-8")) <= 255


# Reject invalid active counts before parsing any measurement files.
@pytest.mark.parametrize(
    "active, error",
    [
        (0, ValueError),
        (True, TypeError),
        (1.5, TypeError),
        (float("nan"), TypeError),
        (float("inf"), TypeError),
    ],
    ids=["zero", "bool", "fraction", "nan", "inf"],
)
def test_profile_parser_requires_positive_integer_active(tmp_path, active, error):
    write_csv(tmp_path / "op_summary.csv", "Task Duration(us)", [10, 20])
    with pytest.raises(error, match="active must be a positive integer"):
        read_device_time_us(tmp_path, active)
