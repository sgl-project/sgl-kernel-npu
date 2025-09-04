#!/usr/bin/env python
# coding=utf-8

"""python api for sgl_kernel_npu."""

import os

import setuptools
import torch
import torch_npu
from setuptools import find_namespace_packages, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
from torch_npu.utils.cpp_extension import NpuExtension

os.environ["SOURCE_DATE_EPOCH"] = "0"

current_version = os.getenv("VERSION", "0.1.0")


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


setuptools.setup(
    name="sgl_kernel_npu",
    version=current_version,
    description="python api for sgl_kernel_npu",
    packages=find_namespace_packages(exclude=("tests*",)),
    ext_modules=[NpuExtension("sgl_kernel_npu._C", sources=[])],
    url="https://github.com/sgl-project/sgl-kernel-npu/",
    license="BSD 3 License",
    python_requires=">=3.7",
    package_data={"sgl_kernel_npu": ["lib/**", "VERSION"]},
)
