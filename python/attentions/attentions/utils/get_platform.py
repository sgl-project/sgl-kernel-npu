#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from enum import Enum, auto
import torch_npu
PLATFORM = None


class NPUDevice(Enum):
    UNDEFINED = auto()
    A2 = auto()
    A5 = auto()
    Duo = auto()


def get_npu_device() -> NPUDevice:
    global PLATFORM
    if PLATFORM is None:
        soc = torch_npu.npu.get_device_name()
        if "310" in soc:
            PLATFORM = NPUDevice.Duo
        elif "910" in soc:
            PLATFORM = NPUDevice.A5 if "910_95" in soc else NPUDevice.A2
        elif "950" in soc:
            PLATFORM = NPUDevice.A5
        else:
            PLATFORM = NPUDevice.UNDEFINED
    return PLATFORM