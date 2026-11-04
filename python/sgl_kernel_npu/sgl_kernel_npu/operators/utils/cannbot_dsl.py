"""Load CANNBot-DSL operator sources without importing them at package import time."""

import importlib
from functools import lru_cache


def require_ascend950(*args, **kwargs):
    """Reject devices outside the hardware supported by these DSL samples."""
    import torch

    devices = [
        value.device for value in (*args, *kwargs.values()) if hasattr(value, "device")
    ]
    if any(getattr(device, "type", None) != "npu" for device in devices):
        raise RuntimeError(
            "CANNBot-DSL DeepSeek V4.1 operators require Ascend 950 NPU tensors"
        )
    try:
        current = torch.npu.current_device()
        indices = [
            current if device.index is None else device.index for device in devices
        ]
        if len(set(indices)) > 1:
            raise RuntimeError("CANNBot-DSL inputs must share the same NPU device")
        index = indices[0] if indices else current
        name = torch.npu.get_device_name(index)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("Cannot identify the Ascend 950 NPU device") from error
    if not name.startswith(("Ascend950PR", "Ascend950DT")):
        raise RuntimeError(
            f"CANNBot-DSL DeepSeek V4.1 operators require Ascend 950; got {name}"
        )
    return index


def ascend950_device(*args, **kwargs):
    """Validate the target and enter its context for upstream metadata paths."""
    import torch

    return torch.npu.device(require_ascend950(*args, **kwargs))


@lru_cache(maxsize=None)
def load_cannbot_dsl(name):
    module = importlib.import_module(
        f"sgl_kernel_npu.implementation.cannbot-dsl.attention.{name}"
    )
    return getattr(module, name)
