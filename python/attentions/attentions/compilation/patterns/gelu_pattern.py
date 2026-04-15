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

import torch
from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu


def create(dtype):
    class GELUPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"
        
        @staticmethod
        def inputs():
            hidden_states = torch.empty(2, 2, 2, dtype=dtype, device="meta")
            return [hidden_states]
        
        @staticmethod
        def pattern(hidden_states):
            def func(hidden_states):
                return torch.nn.GELU(approximate="tanh")(hidden_states)
            return func(hidden_states)

        @staticmethod
        def replacement(hidden_states):
            def func(hidden_states):
                return torch_npu.npu_fast_gelu(hidden_states)
            return func(hidden_states)
    return GELUPattern

GELUPatternGroup = [create(torch.bfloat16)]
