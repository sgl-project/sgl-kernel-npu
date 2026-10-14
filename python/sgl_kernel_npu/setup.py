#!/usr/bin/env python
# coding=utf-8

"""python api for sgl_kernel_npu."""

import os
from configparser import ConfigParser
from pathlib import Path

import setuptools
from build_tools.target_provider import stage_target_providers
from setuptools import find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from torch_npu.utils.cpp_extension import NpuExtension

BUILD_TARGET_ENV = "SGL_KERNEL_NPU_BUILD_TARGET"


class TargetBuildPy(build_py):
    """Stage one target-specific operator provider tree into the wheel.

    Provider selection is a build-time decision driven by
    ``SGL_KERNEL_NPU_BUILD_TARGET``: the relative module path under
    ``target_providers/<target>/`` is the registration key, so the build
    system knows targets but not operators. The source tree deliberately
    ships no provider module -- a source-tree or editable install must fail
    loudly rather than silently default to one target's operator.
    """

    def run(self):
        super().run()

        target = os.environ.get(BUILD_TARGET_ENV)
        if not target:
            raise RuntimeError(
                f"{BUILD_TARGET_ENV} must be set when building the wheel"
            )

        stage_target_providers(
            source_root=WORKING_DIR,
            build_lib=Path(self.build_lib),
            target=target,
        )


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
    packages=find_namespace_packages(
        exclude=("tests*", "target_providers", "target_providers.*")
    ),
    ext_modules=[NpuExtension("sgl_kernel_npu._C", sources=[])],
    cmdclass={"build_py": TargetBuildPy},
    url="https://github.com/sgl-project/sgl-kernel-npu/",
    license="BSD 3 License",
    python_requires=">=3.7",
    package_data={"sgl_kernel_npu": ["lib/**", "VERSION"]},
)
