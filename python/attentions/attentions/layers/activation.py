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

import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from ..utils import ParametersInvalid


class GELU(nn.Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, hidden_states):
        if self.approximate == "none" or self.approximate == "tanh":
            return F.gelu(hidden_states, approximate=self.approximate)
        elif self.approximate == "fast":
            return torch_npu.npu_fast_gelu(hidden_states)
        else:
            raise ParametersInvalid(f"The approximate only support 'none', 'tanh' and 'fast', "
                                    f"but got {self.approximate}")


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": GELU(),
    "relu": nn.ReLU(),
    "gelu-tanh": GELU(approximate="tanh"),
    "gelu-fast": GELU(approximate="fast")
}


def get_activation_layer(act_type: str) -> nn.Module:
    """
    Get activation function act_type.

    Args:
        act_type (str): Name of activation function.
            Support 'swish', 'silu', 'mish', 'gelu', 'relu', 'gelu-tanh', 'gelu-fast'.

    Returns:
        nn.Module: Activation function.
    """

    act_type = act_type.lower()
    if act_type in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_type]
    else:
        raise ParametersInvalid(f"Unsupported activation function: {act_type}")