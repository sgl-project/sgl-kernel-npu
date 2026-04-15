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
import torch

from .common import AttentionParam
from .attention_func import (
    attention_math, get_attention_function, get_attention_function_static, get_attention_function_runtime)
from ...utils.exception import ParametersInvalid


def attention_forward(query, key, value, attn_mask=None, scale=None, fused=True, head_first=False, **kwargs):
    """
    Attention forward function for npu. Input layout must be 'BSND' or 'BNSD'.
    Args:
        query ('torch.Tensor'):
            The input query of attention calculation formula.
        key ('torch.Tensor'):
            The input key of attention calculation formula.
        value ('torch.Tensor'):
            The input value of attention calculation formula.
        attn_mask ('torch.Tensor', *optional*, defaults to `None`):
            The input attn_mask of attention calculation formula.
        scale ('float', *optional*, defaults to `None`):
            The input scale of attention calculation formula.
        fused ('bool', *optional*, defaults to `True`):
            Whether to use the fusion operator. Set 'False' to use original calculation.
        head_first (bool):
            In the layout of q k v, if N is before S, set to True; otherwise, set to False.
        kwargs:
            opt_mode ('str', *optional*, defaults to `runtime`):
                The mode to dispatch fused op. Only takes effect when fused is set to 'True'.
                Only supports: 'runtime', 'static', 'manual'.
                runtime: Dynamically search for the optimal operator at runtime. Only the first search will take time.
                static: Obtain the optimal operator through static table lookup.
                manual: Manually setting the fusion operator type.
            op_type ('str'): Operator type, supports 'prompt_flash_attn', 'fused_attn_score', 'ascend_laser_attention'.
                Only takes effect when opt_mode is set to 'manual'.
            layout ('str'): Operator layout, supports 'BNSD', 'BSND', 'BSH'.
                Only takes effect when opt_mode is set to 'manual'.
    """

    input_params = (query, key, value, attn_mask, scale, fused)
    check_input_params(input_params)
    if not head_first:
        attn_param = AttentionParam(
            query.shape[0], query.shape[-2], query.shape[-1], query.shape[1], key.shape[1], query.dtype, head_first)
    else:
        attn_param = AttentionParam(
            query.shape[0], query.shape[1], query.shape[-1], query.shape[2], key.shape[2], query.dtype, head_first)
    if scale is None:
        scale = attn_param.head_dim ** -0.5
    if not fused:
        return attention_math(query, key, value, attn_mask, scale, head_first)

    opt_mode = kwargs.get("opt_mode", "runtime")

    if opt_mode == "static":
        attn_func = get_attention_function_static(attn_param)
    elif opt_mode == "manual":
        supported_fa_types = {"prompt_flash_attn", "fused_attn_score", "ascend_laser_attention"}
        op_type_env = os.getenv("MINDIE_SD_FA_TYPE")
        op_type = op_type_env or kwargs.get("op_type", "fused_attn_score")
        if op_type not in supported_fa_types:
            raise ParametersInvalid(
                f"Unsupported FA type: '{op_type}'. "
                f"Supported values: {supported_fa_types}")
        layout = kwargs.get("layout", "BNSD")

        attn_func = get_attention_function(attn_param, op_type, layout)
    elif opt_mode == "runtime":
        attn_func = get_attention_function_runtime(attn_param, query, key, value, attn_mask, scale)
    else:
        raise ParametersInvalid(f"The input 'opt_mode':{opt_mode} is invalid. "
            f"The list of supported options is ['runtime', 'static', 'manual']")
    return attn_func(query, key, value, attn_mask, scale)


def check_input_params(input_params):
    query, key, value, attn_mask, scale, fused = input_params
    if not isinstance(query, torch.Tensor):
        raise ParametersInvalid(f"The data type of input query must be torch.Tensor, but got {type(query)}.")
    if not isinstance(key, torch.Tensor):
        raise ParametersInvalid(f"The data type of input key must be torch.Tensor, but got {type(key)}.")
    if not isinstance(value, torch.Tensor):
        raise ParametersInvalid(f"The data type of input value must be torch.Tensor, but got {type(value)}.")
    if query.dim() != 4:
        raise ParametersInvalid(f"The dimensional of input query must be 4, but got {query.dim()}.")
    if key.dim() != 4:
        raise ParametersInvalid(f"The dimensional of input key must be 4, but got {key.dim()}.")
    if value.dim() != 4:
        raise ParametersInvalid(f"The dimensional of input value must be 4, but got {value.dim()}.")
    if not isinstance(fused, bool):
        raise ParametersInvalid(f"The data type of input fused must be bool, but got {type(fused)}.")
    if attn_mask is not None and not isinstance(attn_mask, torch.Tensor):
        raise ParametersInvalid(f"The data type of input attn_mask must be torch.Tensor, but got {type(attn_mask)}.")
    if scale is not None and not isinstance(scale, float):
        raise ParametersInvalid(f"The data type of input scale must be float, but got {type(scale)}.")
