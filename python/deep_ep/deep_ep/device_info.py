"""Device architecture detection utilities for NPU.

The first return value of ``acl.rt.get_device_info(0, 601)`` gives the SoC
version code.  A lookup table maps each known code to an architecture family
("A5" or "A2/A3") so that callers can select the correct quantization path
without hard-coding version checks everywhere.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# SoC version code -> architecture family.
# A5 family (Ascend910C / C310 / 950DT):  9301, 9201, 3510
# A2/A3:                                  2201
# V100:                                    1001
# V200:                                    2002
# V300:                                    3002
DEVICE_VERSION_TABLE = {
    9301: "A5",
    9201: "A5",
    3510: "A5",
    2201: "A2/A3",
    1001: "V100",
    2002: "V200",
    3002: "V300",
}

# All A5 version codes (convenience set)
A5_VERSION_CODES = frozenset(
    v for v, arch in DEVICE_VERSION_TABLE.items() if arch == "A5"
)

# Quantization mode lookup table.
# Key: (param_type, version_code) where param_type is one of
#   "use_fp8", "use_mxfp4", "use_mxfp8" and version_code is from DEVICE_VERSION_TABLE.
# Value: the resolved quant_mode string, or None if the combination is not supported.
QUANT_MODE_TABLE = {
    # --- use_fp8 ---
    ("use_fp8", 9301): "pertoken_fp8_e4m3",
    ("use_fp8", 9201): "pertoken_fp8_e4m3",
    ("use_fp8", 3510): "pertoken_fp8_e4m3",
    ("use_fp8", 2201): "int8",
    ("use_fp8", 1001): "int8",
    ("use_fp8", 2002): "int8",
    ("use_fp8", 3002): "int8",
    # --- use_mxfp4 (A5 only) ---
    ("use_mxfp4", 9301): "mx_fp4_e2m1",
    ("use_mxfp4", 9201): "mx_fp4_e2m1",
    ("use_mxfp4", 3510): "mx_fp4_e2m1",
    ("use_mxfp4", 2201): None,
    ("use_mxfp4", 1001): None,
    ("use_mxfp4", 2002): None,
    ("use_mxfp4", 3002): None,
    # --- use_mxfp8 (A5 only) ---
    ("use_mxfp8", 9301): "mx_fp8_e4m3",
    ("use_mxfp8", 9201): "mx_fp8_e4m3",
    ("use_mxfp8", 3510): "mx_fp8_e4m3",
    ("use_mxfp8", 2201): None,
    ("use_mxfp8", 1001): None,
    ("use_mxfp8", 2002): None,
    ("use_mxfp8", 3002): None,
}

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


def lookup_quant_mode(param_type: str, version_code: int) -> Optional[str]:
    """Look up the quantization mode for a given (param_type, version_code) pair.

    Arguments:
        param_type: one of ``"use_fp8"``, ``"use_mxfp4"``, ``"use_mxfp8"``.
        version_code: SoC version code from ``DEVICE_VERSION_TABLE``.

    Returns:
        The resolved quant_mode string, or ``None`` if the combination is
        not supported on this hardware.
    """
    return QUANT_MODE_TABLE.get((param_type, version_code))
