import logging
import os
import shutil
from pathlib import Path
import sysconfig

import setuptools
from setuptools import setup

import torch
import torch_npu


logger = logging.getLogger(__name__)


def _find_ascend_home():
    """
    Find the ASCEND toolkit home directory.
    It prioritizes the ASCEND_TOOLKIT_HOME environment variable.
    If not set, it falls back to the common default installation path:
    /usr/local/Ascend/ascend-toolkit/latest
    """
    home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if home:
        return home
    default_home = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.isdir(default_home):
        return default_home
    maybe = "/usr/local/Ascend/ascend-toolkit"
    latest = os.path.join(maybe, "latest")
    return latest if os.path.isdir(latest) else default_home


def _find_sheme_home():
    home = os.environ.get("SHMEM_HOME_PATH")
    if home:
        return home
    default_home = "/usr/local/Ascend/shmem/latest"
    return default_home


def _find_python_include():
    return sysconfig.get_path('include')


ascend_home = Path(_find_ascend_home()).resolve()
shmem_home = Path(_find_sheme_home()).resolve()
python_include_dir = Path(_find_python_include()).resolve()
torch_dir = Path(os.path.dirname(torch.__file__)).resolve()
torch_npu_dir = Path(os.path.dirname(torch_npu.__file__)).resolve()
repo_root = Path(__file__).resolve().parents[3]  # sgl-kernel-npu/


include_dirs = [
    str(python_include_dir),
    str((ascend_home / "include").resolve()),
    str((torch_npu_dir / "include").resolve()),
    str((torch_dir / "include").resolve()),
    str((torch_dir / "include/torch/csrc/api/include").resolve()),
    str((shmem_home / "shmem/include").resolve()),
    str((repo_root / "contrib/shmem_allocator").resolve())
]

library_dirs = [
    str((torch_dir / "lib").resolve()),
    str((torch_npu_dir / "lib").resolve()),
    str((shmem_home / "shmem/lib").resolve())
]

logger.warning(f"Using ASCEND_TOOLKIT_HOME at: {ascend_home}")
logger.warning(f"Using SHMEM_HOME_PATH at: {shmem_home}")
logger.warning(f"Include dirs: {include_dirs}")
logger.warning(f"Library dirs: {library_dirs}")


extra_compile_args = ["-std=c++17", "-hno-unused-parameter", "-lno-unused-function", "-Wunused-value", "-Wcast-align",
                      "-Wcast-qual", "-Winvalid-pch", "-Wwrite-strings", "-Wsign-compare", "-Wextra",
                      "-O3", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fstack-protector-strong",
                      "-Wl,-z,noexecstack", "-Wl,-z,relro", "-Wl,-z,now", "-fPIE", "-fPIC", "-ftrapv", "-s"]

common_macros = []

csrc_dir = repo_root / "contrib" / "shmem_allocator" / "csrc"
setup(
    name="shmem_allocator",
    version="0.0.1",
    ext_modules=[
        setuptools.Extension(
            "shmem_allocator.lib.libshmem_allocator",
            sources=[
                str(csrc_dir / "NPUCommon.cpp"),
                str(csrc_dir / "NPUShmemAllocator.cpp"),
                str(csrc_dir / "shmem_alloc_mm_heap.cpp")
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            # CUDA -> ACL
            libraries=["torch", "torch_npu", "shmem"],
            define_macros=[
                *common_macros,
            ],
            extra_compile_args=extra_compile_args,
            py_limited_api=True,
            language="c++"
        )
    ],
    python_requires=">=3.9",
    packages=setuptools.find_packages(
        include=["shmem_allocator", "shmem_allocator.*"]
    ),
)
