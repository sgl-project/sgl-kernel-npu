"""Hardware generation detection for the operator registry."""

import enum


class Generation(enum.Enum):
    """Hardware generations the registry can declare support for."""

    ARCH22 = "arch22"
    ARCH35 = "arch35"


# Architecture IDs reported by acl.rt.get_device_info(device, 601).
ARCH_TO_GENERATION = {
    2201: Generation.ARCH22,
    3510: Generation.ARCH35,
}


class HardwareDetectionError(RuntimeError):
    """The current device architecture could not be determined."""


class Hardware:
    """Detected device generation, cached for the process lifetime."""

    def __init__(self, arch_id):
        self.arch_id = arch_id
        self.generation = ARCH_TO_GENERATION[arch_id]

    def __repr__(self):
        return "Hardware(arch_id={}, generation={})".format(self.arch_id, self.generation.value)


def get_hardware():
    """Detect the architecture of visible device 0 via ACL.

    Importing the registry performs this once; selecting a different device
    later does not change the detected generation.
    """
    try:
        import acl
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise HardwareDetectionError("cannot import acl to detect hardware: {}".format(exc)) from exc

    ret = acl.rt.get_device_info(0, 601)
    value, status = ret if isinstance(ret, tuple) else (ret, 0)
    if status != 0:
        raise HardwareDetectionError("acl.rt.get_device_info failed with status {}".format(status))
    if value not in ARCH_TO_GENERATION:
        raise HardwareDetectionError(
            "unknown architecture id {!r}; known: {}".format(value, sorted(ARCH_TO_GENERATION))
        )
    return Hardware(value)
