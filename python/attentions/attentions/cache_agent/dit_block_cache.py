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

from typing import List, Tuple
from laser_attention.utils.logs.logging import logger
from laser_attention.utils.exception import ParametersInvalid, ModelExecError
from .cache import CacheConfig, CacheBase


class DiTBlockCache(CacheBase):
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache = [None] * 2  # Cache two result, 1th: hidden_states, 2th: encoder_hidden_states
        self._time_cache = {}
        self._output_count = 1

    def apply_imp(self, func: callable, *args, **kwargs):
        if "hidden_states" not in kwargs:
            raise ParametersInvalid("[DiTBlockCache]: Cannot find 'hidden_states' in kwargs.")
        hidden_states = kwargs.pop("hidden_states")
        if hidden_states is None:
            raise ParametersInvalid("[DiTBlockCache]: Input 'hidden_states' is None.")
        
        # update cache
        if not self._use_cache():
            if "encoder_hidden_states" not in kwargs:
                encoder_hidden_states = None
                res = func(hidden_states, *args, **kwargs)
            else:
                encoder_hidden_states = kwargs.pop("encoder_hidden_states")
                res = func(hidden_states, encoder_hidden_states, *args, **kwargs)
            self._update_cache(res, hidden_states, encoder_hidden_states)
            return res

        # reuse cache
        res = self._get_cache()
        if self._output_count == 2:
            if "encoder_hidden_states" not in kwargs:
                raise ParametersInvalid("[DiTBlockCache] 'encoder_hidden_states' is required "
                    "when the output count of cache function is 2.")
            encoder_hidden_states = kwargs.pop("encoder_hidden_states")
            return hidden_states + res[0], encoder_hidden_states + res[1]

        return hidden_states + res[0]
    
    def _use_cache(self):
        if self._cur_step < self._config.step_start:
            return False
        else:
            diftime = self._cur_step - self._config.step_start
            if diftime not in self._time_cache:
                self._time_cache[diftime] = diftime % self._config.step_interval == 0
            if self._time_cache[diftime]:
                return False
            elif self._cur_block < self._config.block_start or self._cur_block >= self._config.block_end:
                return False
            else:
                return True

    def _get_cache(self):
        logger.debug(f"[DiTBlockCache] step: {self._cur_step} block: {self._cur_block} reuse cache.")
        if self._cur_block == self._config.block_start:
            return self._cache
        else:
            return [0, 0]

    def _update_cache(self, res, ori_hidden_states, ori_encoder_hidden_states):
        diftime = self._cur_step - self._config.step_start
        if not (self._cur_step >= self._config.step_start and self._time_cache[diftime]):
            return
        
        # update output count
        self._output_count = len(res) if isinstance(res, (List, Tuple)) else 1

        if self._output_count > 2 or self._output_count < 1:
            raise ModelExecError(
                f"[DiTBlockCache] The output count of cache function must be 1 or 2, but got {self._output_count}.")

        if self._cur_block == self._config.block_start:
            logger.debug(f"[DiTBlockCache] step: {self._cur_step} block: {self._cur_block} update cache begin.")
            self._cache = [ori_hidden_states, ori_encoder_hidden_states]
        elif self._cur_block == (self._config.block_end - 1):
            logger.debug(f"[DiTBlockCache] step: {self._cur_step} block: {self._cur_block} update cache end.")
            if self._output_count == 2:
                hidden_states, encoder_hidden_states = res
                if hidden_states is None or encoder_hidden_states is None:
                    raise ModelExecError("[DiTBlockCache] The output of cache function is None.")
                if ori_encoder_hidden_states is None:
                    raise ParametersInvalid("[DiTBlockCache] 'encoder_hidden_states' is required "
                        "when the output count of cache function is 2.")

                self._cache[0] = hidden_states - self._cache[0]
                self._cache[1] = encoder_hidden_states - self._cache[1]
            else:
                if res is None:
                    raise ModelExecError("[DiTBlockCache] The output of cache function is None.")
                self._cache[0] = res - self._cache[0]
                if self._output_count == 1 and ori_encoder_hidden_states is not None:
                    logger.debug("[DiTBlockCache] Cache function only got one output while input "
                        "'encoder_hidden_states' is not None, cache 'encoder_hidden_states' will be ignored.")

    def _release(self):
        self._cache = [None] * 2
        