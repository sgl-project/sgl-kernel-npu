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
import torch.nn as nn
import torch_npu
from ..utils.exception import ParametersInvalid
from . import _custom_ops as ops


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, if_fused=True):
        if hidden_states.dim() < 2 or hidden_states.dim() > 8:
            raise ParametersInvalid("The input dimension should be greater than 1 and less than 9.")
        if if_fused:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


def check_input_params(layernorm, x, impl_mode, fused):
    if not isinstance(layernorm, torch.nn.LayerNorm):
        raise ParametersInvalid(f"The type of input layernorm must be torch.nn.LayerNorm, but got {type(layernorm)}.")
    if not isinstance(fused, bool):
        raise ParametersInvalid(f"The data type of input fused must be bool, but got {type(fused)}.")
    if impl_mode not in [0, 1, 2]:
        raise ParametersInvalid(f"Expected impl_mode to be in [0, 1, 2], but now got [{impl_mode}]")
    if impl_mode == 2:
        if not (
            x.dtype == torch.float16
            and (layernorm.weight is None or layernorm.weight.dtype == torch.float16)
            and (layernorm.bias is None or layernorm.bias.dtype == torch.float16)
        ):
            raise ParametersInvalid(f"only support all input dtype float16!")


def fast_layernorm(
    norm: torch.nn.LayerNorm, 
    x: torch.Tensor,
    impl_mode: int = 0,
    fused: bool = True) -> torch.Tensor:
    """
    Args:
        norm (torch.nn.LayerNorm):
            The LayerNorm module.
        x (torch.Tensor):
            Tensor to apply LayerNorm. x must be 3-dimensional.
            The supported layout: [B,S,H].
        impl_mode (int):
            Specifies the compute mode for the kernel. The value must be in [0, 1, 2]. The default value is 0.
            0 indicates the high-precision mode, 1 indicates the high-performance mode, and 2 indicates the
            float16 mode. The float16 mode is supported only when all inputs are float16.
        fused (bool): 
            If fused is True, can enable different layernorm mode by specifying 'impl_mode'.
    """
    check_input_params(norm, x, impl_mode, fused)
    if fused:
        out = ops.layernorm(
            x=x, 
            normalized_shape=list(norm.normalized_shape),
            weight=norm.weight, 
            bias=norm.bias, 
            eps=norm.eps,
            impl_mode=impl_mode
        )[0]
    else:
        out = norm(x)
    return out