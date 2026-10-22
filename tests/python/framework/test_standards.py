"""Accuracy standard validation and per-dtype selection."""

import pytest

from tests.python.utils.standards import AccuracyStandard, get_standard


# Check the independently specified default tolerances for each supported dtype.
@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("torch.float32", (1e-5, 1e-5)),
        ("torch.float16", (1e-3, 1e-3)),
        ("torch.bfloat16", (5e-2, 5e-2)),
        ("torch.int8", (0, 0)),
        ("torch.int32", (0, 0)),
        ("torch.int64", (0, 0)),
    ],
)
def test_default_standards_match_dtype_policy(dtype, expected):
    standard = get_standard(dtype)
    assert (standard.atol, standard.rtol) == expected


# Apply a local dtype override without changing fallback or global defaults.
def test_override_changes_only_requested_dtype():
    custom = AccuracyStandard(atol=0, rtol=0.125)
    overrides = {"torch.float32": custom}

    assert get_standard("torch.float32", overrides) is custom
    fallback = get_standard("torch.float16", overrides)
    assert (fallback.atol, fallback.rtol) == (1e-3, 1e-3)
    default = get_standard("torch.float32")
    assert (default.atol, default.rtol) == (1e-5, 1e-5)
    assert overrides == {"torch.float32": custom}


# Reject each invalid tolerance independently in either field.
@pytest.mark.parametrize("field", ["atol", "rtol"])
@pytest.mark.parametrize(
    "value",
    [-0.125, float("nan"), float("inf"), -float("inf")],
    ids=["negative", "nan", "positive-inf", "negative-inf"],
)
def test_invalid_tolerances_are_rejected(field, value):
    values = {"atol": 0, "rtol": 0}
    values[field] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        AccuracyStandard(**values)


# Validate all mapping entries, including invalid entries for unused dtypes.
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param([], id="not-a-mapping"),
        pytest.param({1: AccuracyStandard(0, 0)}, id="non-string-key"),
        pytest.param({"torch.float32": (0, 0)}, id="invalid-value"),
        pytest.param(
            {"torch.float32": AccuracyStandard(0, 0), "unused": object()},
            id="invalid-unused-entry",
        ),
    ],
)
def test_invalid_standard_mapping_is_rejected(overrides):
    with pytest.raises(TypeError, match="map dtype strings to AccuracyStandard"):
        get_standard("torch.float32", overrides)


# Require an explicit override when a dtype has no default standard.
def test_unknown_dtype_has_no_default_standard():
    assert get_standard("torch.float64") is None
    custom = AccuracyStandard(atol=0, rtol=0)
    assert get_standard("torch.float64", {"torch.float64": custom}) is custom
