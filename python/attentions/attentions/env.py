#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
from pathlib import Path


def set_environment_variables():
    current_script_dir = Path(__file__).resolve().parent
    custom_op_path1 = os.path.join(current_script_dir, "ops/vendors/aie_ascendc")
    custom_op_path2 = os.path.join(current_script_dir, "ops/vendors/customize")
    old_custom_op_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    new_custom_op_path = f"{custom_op_path1}:{custom_op_path2}:{old_custom_op_path}"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = new_custom_op_path
