import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu


def _prepend_env_path(name, path):
    entries = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if path not in entries:
        entries.insert(0, path)
        os.environ[name] = ":".join(entries)


def _configure_custom_opp():
    package_path = pathlib.Path(__file__).resolve().parent
    vendor_path = package_path / "vendors" / "sgl_kernel_npu"
    if vendor_path.is_dir():
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", str(vendor_path))


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_configure_custom_opp()
_load_sgl_kernel_npu()
