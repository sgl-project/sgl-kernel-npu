import logging
from enum import Enum
from functools import lru_cache

import torch_npu

logger = logging.getLogger(__name__)


class NpuDeviceFamily(Enum):
    """Ascend device families relevant to kernel implementation selection."""

    ASCEND_910B = "ascend_910b"
    ASCEND_910C = "ascend_910c"
    ASCEND_950 = "ascend_950"
    UNKNOWN = "unknown"


def _family_from_soc_version(soc_version: int) -> NpuDeviceFamily:
    if 220 <= soc_version <= 225:
        return NpuDeviceFamily.ASCEND_910B
    if 250 <= soc_version <= 255:
        return NpuDeviceFamily.ASCEND_910C
    if soc_version == 260:
        return NpuDeviceFamily.ASCEND_950
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
