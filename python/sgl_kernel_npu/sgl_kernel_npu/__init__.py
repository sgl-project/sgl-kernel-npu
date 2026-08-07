import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu


def _prepend_env_path(name, path):
    path = str(path)
    current_paths = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    if path in current_paths:
        current_paths.remove(path)
    os.environ[name] = os.pathsep.join([path, *current_paths])


def _setup_bundled_custom_ops():
    package_path = pathlib.Path(__file__).resolve().parent
    vendors_path = package_path / "vendors"
    for vendor_name in ("aie_ascendc", "customize"):
        vendor_path = vendors_path / vendor_name
        if vendor_path.is_dir():
            _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_path)


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).resolve().parent
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_setup_bundled_custom_ops()
_load_sgl_kernel_npu()
