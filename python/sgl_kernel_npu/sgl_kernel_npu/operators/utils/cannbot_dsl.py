"""Load CANNBot-DSL operator sources without importing them at package import time."""

import importlib
import importlib.metadata
from contextlib import contextmanager
from functools import lru_cache
from itertools import chain

# The pinned sources use APIs removed in 0.8.2. Upgrade sources and runtime together.
DSL_VERSION = "0.8.0"
_NO_DEVICE = object()


@lru_cache(maxsize=None)
def _validate_device(device_index):
    """Cache successful hardware checks by logical NPU index; never cache failures."""
    import torch

    name = torch.npu.get_device_name(device_index)
    if not name.startswith(("Ascend950PR", "Ascend950DT")):
        raise RuntimeError(
            f"CANNBot-DSL DeepSeek V4.1 operators require Ascend 950; got {name}"
        )


def require_ascend950(*args, **kwargs):
    """Return the shared input NPU index, or the current NPU for tensor-less calls.

    Check input devices on every call; only the hardware identity is cached.
    """
    import torch

    try:
        target_index = None
        current_index = None
        for value in chain(args, kwargs.values()):
            device = getattr(value, "device", _NO_DEVICE)
            if device is _NO_DEVICE:
                continue
            if getattr(device, "type", None) != "npu":
                raise RuntimeError(
                    "CANNBot-DSL DeepSeek V4.1 operators require Ascend 950 NPU tensors"
                )
            input_index = device.index
            if input_index is None:
                if current_index is None:
                    current_index = torch.npu.current_device()
                input_index = current_index
            if target_index is None:
                target_index = input_index
            elif input_index != target_index:
                raise RuntimeError("CANNBot-DSL inputs must share the same NPU device")
        if target_index is None:
            target_index = torch.npu.current_device()
        _validate_device(target_index)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("Cannot identify the Ascend 950 NPU device") from error
    return target_index


@contextmanager
def ascend950_device(*args, **kwargs):
    """Validate the target and enter its context for upstream metadata paths."""
    import torch

    target_index = require_ascend950(*args, **kwargs)
    if target_index == torch.npu.current_device():
        yield
    else:
        with torch.npu.device(target_index):
            yield


@lru_cache(maxsize=1)
def initialize_backend():
    """Check the source-compatible JIT runtime once, before importing any operator."""
    try:
        version = importlib.metadata.version("cannbot-dsl")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Install cannbot-dsl=={DSL_VERSION} in the matching Linux/Python environment"
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
