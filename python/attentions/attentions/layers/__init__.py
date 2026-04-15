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

from .flash_attn.attention_forward import attention_forward
from .flash_attn.attention_forward_varlen import attention_forward_varlen
from .flash_attn.sparse_flash_attn import sparse_attention
from .activation import get_activation_layer
from .norm import RMSNorm, fast_layernorm
from .adalayernorm import layernorm_scale_shift
from .rope import rotary_position_embedding