"""Resolve migrated CANNBot-DSL sample modules for the upstream precision tests."""

import importlib
from pathlib import Path


def load_sample(relative_path):
    name = Path(relative_path).stem
    return importlib.import_module(
        f"sgl_kernel_npu.implementation.cannbot-dsl.attention.{name}"
    )
