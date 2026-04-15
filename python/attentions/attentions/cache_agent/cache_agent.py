#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# sgl-kernel-npu/svoloch2940194 is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from laser_attention.utils.exception import ConfigError, ParametersInvalid
from laser_attention.utils.logs.logging import logger
from .cache import CacheConfig
from .attention_cache import AttentionCache
from .dit_block_cache import DiTBlockCache


CACHE_METHOD = {
    "attention_cache": AttentionCache,
    "dit_block_cache": DiTBlockCache
}


class CacheAgent:
    def __init__(self, config: CacheConfig):
        self._config = config
        self._check_config()

        self._cache_method = CACHE_METHOD.get(self._config.method)(self._config)
    
    def apply(self, function: callable, *args, **kwargs):
        if not callable(function):
            raise ParametersInvalid("Input function must be callable.")
        
        # If start step[0,1,2,...] >= stpes_count(), not use cache
        if self._config.step_start >= self._config.steps_count or \
            self._config.step_end == self._config.step_start:
            return function(*args, **kwargs)

        # If start block[0,1,2,...] >= blocks_count(), not use cache
        if self._config.block_start >= self._config.blocks_count or \
            self._config.block_end == self._config.block_start:
            return function(*args, **kwargs)
        
        if self._config.step_interval == 1:
            return function(*args, **kwargs)

        return self._cache_method.apply(function, *args, **kwargs)

    def _check_config(self):
        if self._config.method not in CACHE_METHOD.keys():
            raise ConfigError(f"Method '{self._config.method}' is not supported, "
                              f"the list of support methods is {CACHE_METHOD.keys()}.")
        
        if self._config.blocks_count <= 0:
            raise ConfigError(f"The 'blocks_count' in config must > 0, but got {self._config.blocks_count}.")
        if self._config.steps_count <= 0:
            raise ConfigError(f"The 'steps_count' in config must > 0, but got {self._config.steps_count}.")
        
        if self._config.step_start < 0:
            raise ConfigError(f"The 'step_start' in config must >= 0, but got {self._config.step_start}.")
        if self._config.step_interval <= 0:
            raise ConfigError(f"The 'step_interval' in config must > 0, but got {self._config.step_interval}.")
        if self._config.block_start < 0:
            raise ConfigError(f"The 'block_start' in config must >= 0, but got {self._config.block_start}.")

        if self._config.step_end < self._config.step_start:
            raise ConfigError(f"The 'step_end' must >= 'step_start', "
                f"but got {self._config.step_end} and {self._config.step_start}.")
        if self._config.block_end < self._config.block_start:
            raise ConfigError(f"The 'block_end' must >= 'block_start', "
                f"but got {self._config.block_end} and {self._config.block_start}.")
        if self._config.method == "dit_block_cache":
            if self._config.block_end > self._config.blocks_count:
                raise ConfigError(f"The 'block_end' must <= 'blocks_count', "
                    f"but got {self._config.block_end} and {self._config.blocks_count}.")
        if self._config.step_start >= self._config.steps_count or \
            self._config.step_end == self._config.step_start:
            logger.debug(f"'step_start' >= 'steps_count' or 'step_end' == 'step_start', do not apply cache function.")
        if self._config.block_start >= self._config.blocks_count or \
            self._config.block_end == self._config.block_start:
            logger.debug("'block_start' >= 'blocks_count' or 'block_end' == 'block_start', "
                         "do not apply cache function.")
        if self._config.step_interval == 1:
            logger.debug("'step_interval' is 1, do not apply cache function.")