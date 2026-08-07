import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu


def _prepend_env_path(name, path):
    path = str(path)
    current_paths = [item for item in os.environ.get(name, "").split(":") if item]
    if path in current_paths:
        current_paths.remove(path)
    os.environ[name] = ":".join([path, *current_paths])


def _setup_bundled_custom_ops():
    package_path = pathlib.Path(__file__).resolve().parent
    vendor_path = package_path / "vendors" / "customize"
    if not vendor_path.is_dir():
        return

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_path)
    _prepend_env_path("LD_LIBRARY_PATH", vendor_path / "op_api" / "lib")


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).resolve().parent
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_setup_bundled_custom_ops()
_load_sgl_kernel_npu()
