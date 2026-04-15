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

import os
from pathlib import Path

import torch
from ..utils import ParametersInvalid, file_utils
from ..utils.get_platform import get_npu_device, NPUDevice
from . import _custom_ops as ops

npu_device = get_npu_device()


def check_input_params(layernorm, x, scale, shift, fused):
    if not isinstance(layernorm, torch.nn.LayerNorm):
        raise ParametersInvalid(f"The type of input layernorm must be torch.nn.LayerNorm, but got {type(layernorm)}.")
    if not isinstance(x, torch.Tensor):
        raise ParametersInvalid(f"The data type of input x must be torch.Tensor, but got {type(x)}.")
    if not isinstance(scale, torch.Tensor):
        raise ParametersInvalid(f"The data type of input scale must be torch.Tensor, but got {type(scale)}.")
    if not isinstance(shift, torch.Tensor):
        raise ParametersInvalid(f"The data type of input shift must be torch.Tensor, but got {type(shift)}.")
    if not isinstance(fused, bool):
        raise ParametersInvalid(f"The data type of input fused must be bool, but got {type(fused)}.")

    if x.dim() != 3:    # 3: BSH输入dim
        raise ParametersInvalid(f"The dimensional of input x must be a 3, but got {x.dim()}.")
    if scale.dim() not in [2, 3]:    # 2: BH输入dim; 3: B1H输入dim
        raise ParametersInvalid(f"The dimensional of input scale must be a 2 or 3, but got {scale.dim()}.")
    if scale.dim() == 3 and scale.size()[1] != 1:
        raise ParametersInvalid(f"If scale is a 3D tensor, the second dimension must be 1, but got {scale.size()[1]}.")
    if shift.dim() not in [2, 3]:    # 2: BH输入dim; 3: B1H输入dim
        raise ParametersInvalid(f"The dimensional of input shift must be a 2 or 3, but got {shift.dim()}.")
    if shift.dim() == 3 and shift.size()[1] != 1:
        raise ParametersInvalid(f"If shift is a 3D tensor, the second dimension must be 1, but got {shift.size()[1]}.")

    last_dim_x = x.size()[-1]
    last_dim_scale = scale.size()[-1]
    last_dim_shift = shift.size()[-1]
    if last_dim_x != last_dim_scale:
        raise ParametersInvalid(f"The last dimensions of input x and input scale must be equal,  "
                                f"but {last_dim_x} != {last_dim_scale}.")
    if last_dim_scale != last_dim_shift:
        raise ParametersInvalid(f"The last dimensions of input scale and input shift must be equal,  "
                                f"but {last_dim_scale} != {last_dim_shift}.")


def layernorm_scale_shift(
    layernorm: torch.nn.LayerNorm, 
    x: torch.Tensor, 
    scale: torch.Tensor, 
    shift: torch.Tensor, 
    fused: bool = True) -> torch.Tensor:
    """
    Apply AdaLayerNorm to input tensors:
        out = layernorm(x) * (1 + scale) + shift

    Args:
        layernorm (torch.nn.LayerNorm):
            The LayerNorm module.
        x (torch.Tensor):
            Tensor to apply AdaLayerNorm. x must be 3-dimensional.
            The supported layout: [B,S,H].
        scale (torch.Tensor):
            Adaptive Scaling Parameters. scale must be 2 or 3-dimensional.
            The supported layout: [B, H], [B, 1, H].
        shift (torch.Tensor):
            Adaptive offset parameter. shift must be 2 or 3-dimensional.
            The supported layout: [B, H], [B, 1, H].
        fused (bool): 
            If fused is True, using high-performance AdaLayerNorm operator.

    Returns:
        (torch.Tensor): modified tensor with AdaLayerNorm.
    """
    check_input_params(layernorm, x, scale, shift, fused)

    if fused:
        if layernorm.elementwise_affine:
            weight = layernorm.weight
            bias = layernorm.bias
        else:
            weight = None
            bias = None
        if npu_device == NPUDevice.A5:
            out = ops.adaln_v2(
                x=x,
                scale=scale,
                shift=shift,
                weight=weight,
                bias=bias,
                epsilon=layernorm.eps
            )[0]
        else:
            out = ops.adaln(
                x=x, 
                scale=scale, 
                shift=shift,
                weight=weight, 
                bias=bias, 
                epsilon=layernorm.eps
            )
    else:
        if scale.dim() == 2:
            scale = scale[:, None]
        if shift.dim() == 2:
            shift = shift[:, None]
        out = layernorm(x) * (1 + scale) + shift
        
    return out