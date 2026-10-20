#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

#
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:

#     http://license.coscl.org.cn/MulanPSL2

# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import logging
import os
import shutil
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

_native_mode = os.environ.get("ATTENTIONS_BUILD_NATIVE", "1")
if _native_mode not in ("0", "1"):
    raise ValueError("ATTENTIONS_BUILD_NATIVE must be 0 or 1")
BUILD_NATIVE = _native_mode == "1"


def copy_so_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    so_files = [f for f in os.listdir(src_dir) if f.endswith(".so")]
    if not so_files:
        logging.warning(f"No .so files found in {src_dir}")
        return
    for so_file in so_files:
        src_file = os.path.join(src_dir, so_file)
        dest_file = os.path.join(dest_dir, so_file)
        shutil.copy2(src_file, dest_file)
        logging.info(f"Copied {src_file} to {dest_file}")


def ensure_plugin_init():
    plugin_dir = os.path.join(os.getcwd(), "attentions/plugin")
    init_file = os.path.join(plugin_dir, "__init__.py")

    os.makedirs(plugin_dir, exist_ok=True)
    if not os.path.isfile(init_file):
        with open(init_file, "w") as f:
            pass
    else:
        os.remove(init_file)
        with open(init_file, "w") as f:
            pass


class CustomBuildPy(_build_py):
    def run(self):
        # 3. 继续默认 build_py 流程
        super().run()


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = not BUILD_NATIVE


if __name__ == "__main__":
    requirements = ["torch", "torch_npu"]
    if BUILD_NATIVE:
        ensure_plugin_init()
    else:
        requirements.append("mindiesd")

    setup(
        name="attentions",
        version=0.2,
        author="???",
        description="build wheel for laser attention",
        setup_requires=[],
        install_requires=requirements,
        extras_require={"mindiesd": ["mindiesd"]},
        zip_safe=False,
        python_requires=">=3.10",
        include_package_data=BUILD_NATIVE,
        packages=find_packages(
            exclude=(
                ()
                if BUILD_NATIVE
                else (
                    "attentions.plugin",
                    "attentions.plugin.*",
                    "attentions.ops",
                    "attentions.ops.*",
                )
            )
        ),
        package_data={"": ["*.so", "ops/**/*"]} if BUILD_NATIVE else {},
        # Keep a previous native build/lib out of the pure Python wheel.
        options={} if BUILD_NATIVE else {"build": {"build_base": "build/mindiesd"}},
        cmdclass={"build_py": CustomBuildPy, "bdist_wheel": BDistWheel},
    )
