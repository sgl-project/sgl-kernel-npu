#!/usr/bin/env python
# coding=utf-8

"""python api for sgl_kernel_npu."""

import os
import shutil
from configparser import ConfigParser
from pathlib import Path

import setuptools
from setuptools import find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from torch_npu.utils.cpp_extension import NpuExtension

BUILD_TARGET_ENV = "SGL_KERNEL_NPU_BUILD_TARGET"
GEMMA_PROVIDERS = {
    "Ascend910": "_gemma_rmsnorm_native.py",
    "Ascend950": "_gemma_rmsnorm_aclnn.py",
}


class TargetBuildPy(build_py):
    """Stage exactly one Gemma RMSNorm provider as ``norm/gemma_rmsnorm.py``.

    The source tree deliberately ships no ``gemma_rmsnorm.py``: picking the
    provider is a build-time decision, so a source-tree or editable install must
    fail loudly rather than silently default to the 910 operator on A5.
    """

    def run(self):
        super().run()
        target = os.environ.get(BUILD_TARGET_ENV, "Ascend910")
        if target not in GEMMA_PROVIDERS:
            raise ValueError(f"Unsupported wheel target: {target!r}")
        norm_dir = Path(self.build_lib) / "sgl_kernel_npu" / "norm"
        shutil.copyfile(
            norm_dir / GEMMA_PROVIDERS[target], norm_dir / "gemma_rmsnorm.py"
        )
        for provider in GEMMA_PROVIDERS.values():
            (norm_dir / provider).unlink()


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


class Build(build_ext, object):

    def run(self):
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, "build/temp"))
        self.library_dirs.append(os.path.relpath(os.path.join(BASE_DIR, "build/lib")))
        super(Build, self).run()


WORKING_DIR = Path(__file__).resolve().parent
config = ConfigParser()
config.read(WORKING_DIR / "sgl_kernel_npu" / "config.ini")
_version = config.get("global", "version")


setuptools.setup(
    name="sgl_kernel_npu",
    version=_version,
    description="python api for sgl_kernel_npu",
    packages=find_namespace_packages(exclude=("tests*",)),
    ext_modules=[NpuExtension("sgl_kernel_npu._C", sources=[])],
    cmdclass={"build_py": TargetBuildPy},
    url="https://github.com/sgl-project/sgl-kernel-npu/",
    license="BSD 3 License",
    python_requires=">=3.7",
    package_data={"sgl_kernel_npu": ["lib/**", "VERSION"]},
)
