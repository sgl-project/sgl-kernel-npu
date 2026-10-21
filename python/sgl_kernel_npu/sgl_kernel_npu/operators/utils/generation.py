"""Explicit SoC mapping; one hardware generation per process."""

from enum import Enum
from functools import lru_cache
from typing import NamedTuple


class Generation(Enum):
    A2 = "A2"
    A3 = "A3"
    A5 = "A5"


# Exact names only: extend this table when validating additional SoCs.
# Families follow csrc/CMakeLists.txt. Model names follow TritonAscend's
# docs/zh/debug_guide/profiling.md (CANN 9.2 simulator target list).
SOC_TO_GENERATION = {
    **dict.fromkeys((
        "Ascend910B1", "Ascend910B2", "Ascend910B2C", "Ascend910B3",
        "Ascend910B4", "Ascend910B4-1",
    ), Generation.A2),
    **dict.fromkeys((
        "Ascend910_9362", "Ascend910_9372", "Ascend910_9381",
        "Ascend910_9382", "Ascend910_9391", "Ascend910_9392",
    ), Generation.A3),
    **dict.fromkeys((
        "Ascend950", "Ascend950PR", "Ascend950DT",
        "Ascend950PR_950x", "Ascend950PR_950y", "Ascend950PR_950z",
        "Ascend950PR_9571", "Ascend950PR_9572", "Ascend950PR_9573",
        "Ascend950PR_9574", "Ascend950PR_9575", "Ascend950PR_9576",
        "Ascend950PR_9577", "Ascend950PR_9578", "Ascend950PR_957b",
        "Ascend950PR_957c", "Ascend950PR_957d", "Ascend950PR_9581",
        "Ascend950PR_9582", "Ascend950PR_9583", "Ascend950PR_9584",
        "Ascend950PR_9585", "Ascend950PR_9586", "Ascend950PR_9587",
        "Ascend950PR_9588", "Ascend950PR_9589", "Ascend950PR_958a",
        "Ascend950PR_958b", "Ascend950PR_9591", "Ascend950PR_9592",
        "Ascend950PR_9595",
    ), Generation.A5),
}


class HardwareDetectionError(RuntimeError):
    """Hardware could not be identified; this is not an unsupported impl."""


class Hardware(NamedTuple):
    soc_name: str
    generation: Generation


def generation_from_soc(soc_name):
    """Resolve an exact runtime SoC name; never guess unknown hardware."""
    try:
        return SOC_TO_GENERATION[soc_name]
    except (KeyError, TypeError):
        raise HardwareDetectionError(
            f"Unknown SoC {soc_name!r}; update SOC_TO_GENERATION after validation"
        ) from None


@lru_cache(maxsize=1)
def get_hardware():
    """Cache process hardware; registry import calls this before registration.

    All cards on the machine must share a generation. Failures are not cached.
    """
    try:
        import torch
        import torch_npu  # noqa: F401 -- registers torch.npu

        soc_name = torch.npu.get_device_name()
    except Exception as error:
        raise HardwareDetectionError("Failed to identify the current NPU") from error
    return Hardware(soc_name, generation_from_soc(soc_name))
