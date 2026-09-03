"""Device architecture detection utilities for NPU.

The first return value of ``acl.rt.get_device_info(0, 601)`` gives the SoC
version code.  A lookup table maps each known code to an architecture family
("A5" or "A2/A3") so that callers can select the correct quantization path
without hard-coding version checks everywhere.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# SoC version code -> architecture family.
# A5 family (Ascend910C / C310 / 950DT):  9301, 9201, 3510
# A2/A3 family:                            2201, 1001, 2002, 3002
DEVICE_VERSION_TABLE = {
    9301: "A5",
    9201: "A5",
    3510: "A5",
    2201: "A2/A3",
    1001: "A2/A3",
    2002: "A2/A3",
    3002: "A2/A3",
}

# All A5 version codes (convenience set)
A5_VERSION_CODES = frozenset(v for v, arch in DEVICE_VERSION_TABLE.items() if arch == "A5")

_cached_arch: Optional[str] = None


def get_device_version() -> int:
    """Return the SoC version code via the ACL runtime API.

    ``acl.rt.get_device_info(device_id, 601)`` returns a tuple whose first
    element is the version code.
    """
    import acl

    return acl.rt.get_device_info(0, 601)[0]


def get_device_arch() -> str:
    """Detect and cache the current device architecture family.

    Returns ``"A5"`` for Ascend910C / C310 / 950DT devices, or
    ``"A2/A3"`` for other Ascend devices.  On failure, defaults to
    ``"A2/A3"`` so that callers fall back to the safer (int8) path.
    """
    global _cached_arch
    if _cached_arch is not None:
        return _cached_arch

    try:
        version = get_device_version()
        arch = DEVICE_VERSION_TABLE.get(version)
        if arch is None:
            logger.warning(
                "Unknown device version code %s, defaulting to A2/A3.",
                version,
            )
            arch = "A2/A3"
        _cached_arch = arch
    except Exception as e:
        logger.warning(
            "Failed to detect device architecture: %s. Defaulting to A2/A3.",
            e,
        )
        _cached_arch = "A2/A3"

    return _cached_arch


def is_a5_device() -> bool:
    """Return ``True`` when the current device belongs to the A5 family."""
    return get_device_arch() == "A5"