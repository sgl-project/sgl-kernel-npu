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


__all__ = [
    'CacheConfig',
    'CacheAgent',
    'layernorm_scale_shift',
    'attention_forward',
    'attention_forward_varlen',
    'rotary_position_embedding',
    'get_activation_layer',
    'RMSNorm',
    'quantize',
    'TimestepManager',
    'TimestepPolicyConfig',
    'sparse_attention',
    'fast_layernorm'
]


from .env import set_environment_variables
set_environment_variables()

from .cache_agent import CacheConfig, CacheAgent
from .layers import (
    layernorm_scale_shift,
    attention_forward,
    attention_forward_varlen,
    rotary_position_embedding,
    get_activation_layer,
    RMSNorm,
    sparse_attention,
    fast_layernorm
)
from .quantization import quantize, TimestepManager, TimestepPolicyConfig
