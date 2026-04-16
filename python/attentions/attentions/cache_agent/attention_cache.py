#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# 
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from laser_attention.utils.logs.logging import logger
from .cache import CacheConfig, CacheBase


class AttentionCache(CacheBase):
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache = [None] * self._config.blocks_count

    def apply_imp(self, func: callable, *args, **kwargs):
        if self._config.step_start < self._cur_step <= self._config.step_end and \
            ((self._cur_step - self._config.step_start) % self._config.step_interval != 0):
            attn = self._cache[self._cur_block]
            logger.debug(f"[AttentionCache] step: {self._cur_step} block: {self._cur_block} reuse cache.")
        else:
            attn = func(*args, **kwargs)
            if self._config.step_start <= self._cur_step < self._config.step_end:  # 当在step_end之前才缓存
                self._cache[self._cur_block] = attn
                logger.debug(f"[AttentionCache] step: {self._cur_step} block: {self._cur_block} update cache.")
        return attn

    def _release(self):
        self._cache = [None] * self._config.blocks_count
        