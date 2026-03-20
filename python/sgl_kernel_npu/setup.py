#!/usr/bin/env python
# coding=utf-8

"""python api for sgl_kernel_npu."""

import os
from configparser import ConfigParser
from pathlib import Path

import setuptools
from setuptools import find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
from torch_npu.utils.cpp_extension import NpuExtension


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
    url="https://github.com/sgl-project/sgl-kernel-npu/",
    license="BSD 3 License",
    python_requires=">=3.7",
    package_data={"sgl_kernel_npu": ["lib/**", "VERSION"]},
)
