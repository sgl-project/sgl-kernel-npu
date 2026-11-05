"""Load CANNBot-DSL operator sources without importing them at package import time."""

import importlib
import importlib.metadata
from contextlib import contextmanager
from functools import lru_cache
from itertools import chain

DSL_VERSION = "0.8.0"
_NO_DEVICE = object()


@lru_cache(maxsize=None)
def _validate_device(index):
    """Device identity is stable; cache successful validation, never failures."""
    import torch

    name = torch.npu.get_device_name(index)
    if not name.startswith(("Ascend950PR", "Ascend950DT")):
        raise RuntimeError(
            f"CANNBot-DSL DeepSeek V4.1 operators require Ascend 950; got {name}"
        )


def require_ascend950(*args, **kwargs):
    """Reject devices outside the hardware supported by these DSL samples."""
    import torch

    try:
        index = current = None
        for value in chain(args, kwargs.values()):
            device = getattr(value, "device", _NO_DEVICE)
            if device is _NO_DEVICE:
                continue
            if getattr(device, "type", None) != "npu":
                raise RuntimeError(
                    "CANNBot-DSL DeepSeek V4.1 operators require Ascend 950 NPU tensors"
                )
            candidate = device.index
            if candidate is None:
                if current is None:
                    current = torch.npu.current_device()
                candidate = current
            if index is None:
                index = candidate
            elif candidate != index:
                raise RuntimeError("CANNBot-DSL inputs must share the same NPU device")
        if index is None:
            index = torch.npu.current_device()
        _validate_device(index)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("Cannot identify the Ascend 950 NPU device") from error
    return index


@contextmanager
def ascend950_device(*args, **kwargs):
    """Validate the target and enter its context for upstream metadata paths."""
    import torch

    index = require_ascend950(*args, **kwargs)
    if index == torch.npu.current_device():
        yield
    else:
        with torch.npu.device(index):
            yield


@lru_cache(maxsize=1)
def initialize_backend():
    """Validate the optional JIT runtime once."""
    try:
        version = importlib.metadata.version("cannbot-dsl")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install cannbot-dsl==0.8.0 in the matching Linux/Python environment"
        ) from error
    if version != DSL_VERSION:
        raise RuntimeError(f"Expected cannbot-dsl=={DSL_VERSION}; found {version}")


@lru_cache(maxsize=None)
def load_cannbot_dsl(name):
    initialize_backend()
    module = importlib.import_module(
        f"sgl_kernel_npu.implementation.cannbot-dsl.attention.{name}"
    )
    return getattr(module, name)
