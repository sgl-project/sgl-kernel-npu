import logging
from enum import Enum
from functools import lru_cache

import torch_npu

logger = logging.getLogger(__name__)


class NpuDeviceFamily(Enum):
    """Ascend device families relevant to kernel implementation selection."""

    ASCEND_310P = "310p"
    A2 = "a2"
    A3 = "a3"
    A5 = "a5"
    UNKNOWN = "unknown"


def _family_from_soc_version(soc_version: int) -> NpuDeviceFamily:
    if 200 <= soc_version <= 205:
        return NpuDeviceFamily.ASCEND_310P
    if 220 <= soc_version <= 225:
        return NpuDeviceFamily.A2
    if 250 <= soc_version <= 255:
        return NpuDeviceFamily.A3
    if soc_version == 260:
        return NpuDeviceFamily.A5
    return NpuDeviceFamily.UNKNOWN


@lru_cache(maxsize=1)
def get_npu_device_family() -> NpuDeviceFamily:
    """Detect the current Ascend family once, falling back conservatively."""

    try:
        soc_version = torch_npu.npu.get_soc_version()
        return _family_from_soc_version(soc_version)
    except Exception:
        logger.warning(
            "Failed to detect the Ascend SoC version; using conservative kernel "
            "fallbacks.",
            exc_info=True,
        )
        return NpuDeviceFamily.UNKNOWN
