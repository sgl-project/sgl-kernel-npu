"""Initial accuracy thresholds, to be validated on supported NPU generations."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyStandard:
    atol: float
    rtol: float

    def __post_init__(self):
        if any(
            not math.isfinite(value) or value < 0 for value in (self.atol, self.rtol)
        ):
            raise ValueError("atol and rtol must be finite and nonnegative")


# Elementwise |result - result_ref| <= atol + rtol * |result_ref|.
# Integer Tensor defaults require exact comparison.
DEFAULT_STANDARDS = {
    "torch.float32": AccuracyStandard(atol=1e-5, rtol=1e-5),
    "torch.float16": AccuracyStandard(atol=1e-3, rtol=1e-3),
    "torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=5e-2),
    "torch.int8": AccuracyStandard(atol=0, rtol=0),
    "torch.int32": AccuracyStandard(atol=0, rtol=0),
    "torch.int64": AccuracyStandard(atol=0, rtol=0),
}


def get_standard(dtype, standard=None):
    """Resolve a dtype-string override, falling back to the shared defaults."""
    if standard is not None:
        if not isinstance(standard, Mapping):
            raise TypeError(
                "standard must map dtype strings to AccuracyStandard values"
            )
        for key, value in standard.items():
            if not isinstance(key, str) or not isinstance(value, AccuracyStandard):
                raise TypeError(
                    "standard must map dtype strings to AccuracyStandard values"
                )
        if dtype in standard:
            return standard[dtype]
    return DEFAULT_STANDARDS.get(dtype)
