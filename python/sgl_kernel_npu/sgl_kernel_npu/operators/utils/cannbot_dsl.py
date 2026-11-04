"""Load CANNBot-DSL operator sources without importing them at package import time."""

import importlib
import importlib.metadata
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

DSL_VERSION = "0.8.0"
AICPU_LIBRARIES = (
    "mqsmla_metadata_kernel.so",
    "metadata_kernel.so",
    "qsli_metadata_kernel.so",
)


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

    devices = [
        value.device for value in (*args, *kwargs.values()) if hasattr(value, "device")
    ]
    if any(getattr(device, "type", None) != "npu" for device in devices):
        raise RuntimeError(
            "CANNBot-DSL DeepSeek V4.1 operators require Ascend 950 NPU tensors"
        )
    try:
        current = (
            torch.npu.current_device()
            if not devices or any(device.index is None for device in devices)
            else None
        )
        indices = [
            current if device.index is None else device.index for device in devices
        ]
        if len(set(indices)) > 1:
            raise RuntimeError("CANNBot-DSL inputs must share the same NPU device")
        index = indices[0] if indices else current
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
    """Validate the optional runtime and register a bundled Native directory once."""
    try:
        version = importlib.metadata.version("cannbot-dsl")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install cannbot-dsl==0.8.0 in the matching Linux/Python environment"
        ) from error
    if version != DSL_VERSION:
        raise RuntimeError(f"Expected cannbot-dsl=={DSL_VERSION}; found {version}")
    source = (
        Path(__file__).resolve().parents[2] / "implementation/cannbot-dsl/attention"
    )
    native = source / "_native"
    if native.is_dir():
        missing = [
            name for name in AICPU_LIBRARIES if not (source / "_aicpu" / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"Incomplete CANNBot-DSL AICPU bundle: {missing}")
        from cannbotdsl.aot import register_directory

        register_directory(native)
    elif os.environ.get("CANNBOTDSL_NATIVE_BINARY_MODE") == "require":
        raise RuntimeError(
            "This wheel has no CANNBot-DSL Native bundle; build with --with-cannbot-dsl"
        )


@lru_cache(maxsize=None)
def load_cannbot_dsl(name):
    initialize_backend()
    module = importlib.import_module(
        f"sgl_kernel_npu.implementation.cannbot-dsl.attention.{name}"
    )
    return getattr(module, name)
