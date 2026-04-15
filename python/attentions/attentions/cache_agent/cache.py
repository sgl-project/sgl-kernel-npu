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

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CacheConfig:
    method: str
    blocks_count: int
    steps_count: int
    step_start: int = 0
    step_interval: int = 1
    step_end: int = 10000
    block_start: int = 0
    block_end: int = 10000


class CacheBase(ABC):
    def __init__(self, config: CacheConfig):
        super().__init__()
        self._config = config
        self._cur_step = 0
        self._cur_block = 0
    
    
    def apply(self, func: callable, *args, **kwargs):
        res = self.apply_imp(func, *args, **kwargs)
        self._counter()  # 内部计数
        return res

    @abstractmethod
    def apply_imp(self, func: callable, *args, **kwargs):
        pass

    @abstractmethod
    def _release(self):
        pass

    def _counter(self):
        self._cur_block += 1  # 内部计数+1
        if self._cur_block == self._config.blocks_count:
            self._cur_step += 1  # 满足blocks_count时，step+1
            self._cur_block = 0
            if self._cur_step == self._config.steps_count:
                self._cur_step = 0  # 满足到step_count的时候清0，并清空内存
                self._release()