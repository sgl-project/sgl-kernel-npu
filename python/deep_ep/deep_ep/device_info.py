"""Device architecture detection utilities for NPU."""

from typing import Optional

DEVICE_VERSION_TABLE = {
    9301: "A5",
    9201: "A5",
    3510: "A5",
    2201: "A2/A3",
    1001: "V100",
    2002: "V200",
    3002: "V300",
}


QUANT_MODE_TABLE = {
    ("use_fp8", 9301): "pertoken_fp8_e4m3",
    ("use_fp8", 9201): "pertoken_fp8_e4m3",
    ("use_fp8", 3510): "pertoken_fp8_e4m3",
    ("use_fp8", 2201): "int8",
    ("use_fp8", 1001): "int8",
    ("use_fp8", 2002): "int8",
    ("use_fp8", 3002): "int8",
    ("use_mxfp4", 9301): "mx_fp4_e2m1",
    ("use_mxfp4", 9201): "mx_fp4_e2m1",
    ("use_mxfp4", 3510): "mx_fp4_e2m1",
    ("use_mxfp4", 2201): None,
    ("use_mxfp4", 1001): None,
    ("use_mxfp4", 2002): None,
    ("use_mxfp4", 3002): None,
    ("use_mxfp8", 9301): "mx_fp8_e4m3",
    ("use_mxfp8", 9201): "mx_fp8_e4m3",
    ("use_mxfp8", 3510): "mx_fp8_e4m3",
    ("use_mxfp8", 2201): None,
    ("use_mxfp8", 1001): None,
    ("use_mxfp8", 2002): None,
    ("use_mxfp8", 3002): None,
}


def get_device_version() -> Optional[int]:
    """Return the SoC version code, or ``None`` when it is unavailable."""
    import acl

    version_code, ret = acl.rt.get_device_info(0, 601)
    if ret != 0 or version_code == 0:
        return None
    return version_code
