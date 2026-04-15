#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

# sgl-kernel-npu/svoloch2940194 is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:

#     http://license.coscl.org.cn/MulanPSL2

# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, 
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import logging
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def copy_so_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    so_files = [f for f in os.listdir(src_dir) if f.endswith('.so')]
    if not so_files:
        logging.warning(f"No .so files found in {src_dir}")
        return
    for so_file in so_files:
        src_file = os.path.join(src_dir, so_file)
        dest_file = os.path.join(dest_dir, so_file)
        shutil.copy2(src_file, dest_file)
        logging.info(f"Copied {src_file} to {dest_file}")


def ensure_plugin_init():
    plugin_dir = os.path.join(os.getcwd(), 'attentions/plugin')
    init_file = os.path.join(plugin_dir, '__init__.py')
    
    os.makedirs(plugin_dir, exist_ok=True)   
    if not os.path.isfile(init_file):
        with open(init_file, 'w') as f:
            pass
    else:
        os.remove(init_file)
        with open(init_file, 'w') as f:
            pass


class CustomBuildPy(_build_py):
    def run(self):        
        # 3. 继续默认 build_py 流程
        super().run()


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # 标记为二进制 wheel，否则会生成 py3-none-any
        self.root_is_pure = False


if __name__ == "__main__":
    requirements = ["torch", "torch_npu"]
    ensure_plugin_init()

    setup(
        name="attentions",
        version=0.2,
        author="???",
        description="build wheel for laser attention",
        setup_requires=[],
        install_requires=requirements,
        zip_safe=False,
        python_requires=">=3.10",
        include_package_data=True,
        packages=find_packages(),
        package_data={
            "": [
                "*.so",  
                "ops/**/*"
            ]
        },
        cmdclass={
            "build_py": CustomBuildPy,
            "bdist_wheel": BDistWheel
        }
    )

