"""Explicit ACL architecture mapping; one NPU architecture per process."""

from enum import Enum
from functools import lru_cache
from typing import NamedTuple


class Generation(Enum):
    ARCH22 = "arch22"
    ARCH35 = "arch35"


ACL_DEV_ATTR_NPU_ARCH = 601
ARCH_TO_GENERATION = {
    2201: Generation.ARCH22,
    3510: Generation.ARCH35,
}
# torch_npu SoC families, used when CANN does not support ACL attribute 601.
SOC_TO_ARCH = {
    220: 2201,
    250: 2201,
    260: 3510,
}


class HardwareDetectionError(RuntimeError):
    """Hardware could not be identified; this is not an unsupported impl."""


class Hardware(NamedTuple):
    arch_id: int
    generation: Generation


def generation_from_arch(arch_id):
    """Resolve an exact ACL architecture ID; never guess unknown hardware."""
    try:
        return ARCH_TO_GENERATION[arch_id]
    except (KeyError, TypeError):
        raise HardwareDetectionError(
            f"Unknown NPU architecture {arch_id!r}; "
            "update ARCH_TO_GENERATION after validation"
        ) from None


def _arch_from_soc():
    """Query the SoC family through torch_npu for older CANN versions."""
    try:
        import torch_npu

        soc = torch_npu._C._npu_get_soc_version()
        return SOC_TO_ARCH[soc // 10 * 10]
    except Exception as error:
        raise HardwareDetectionError(
            f"Failed to query NPU architecture through torch_npu: {error}"
        ) from error


@lru_cache(maxsize=1)
def get_hardware():
    """Cache process hardware; registry import calls this before registration.

    All cards must share an architecture. Query visible device 0, falling back
    to torch_npu on CANN versions without ACL attribute 601.
    Failures are not cached.
    """
    try:
        import acl

        arch_id, ret = acl.rt.get_device_info(0, ACL_DEV_ATTR_NPU_ARCH)
    except Exception:
        arch_id = _arch_from_soc()
    else:
        if ret != 0:
            arch_id = _arch_from_soc()
    return Hardware(arch_id, generation_from_arch(arch_id))
