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

from .logs.logging import logger


class ParametersInvalid(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000001] Parameters invalid. {message}"
        logger.error(self.message)
        super().__init__(self.message)


class ConfigError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000002] Config parameter err. {message}"
        logger.error(self.message)
        super().__init__(self.message)


class TorchError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000003] Torch exec err. {message}"
        logger.error(self.message)
        super().__init__(self.message)


class ModelInitError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000004] Model init err. {message}"
        logger.error(self.message)
        super().__init__(self.message)


class ModelExecError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000005] Model exec err. {message}"
        logger.error(self.message)
        super().__init__(self.message)