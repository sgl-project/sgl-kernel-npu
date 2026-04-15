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

from functools import partial
import time

import torch

from .common import AttentionParam, lru_cache_by_attn_param
from .attention_operate import device_duo_op, device_800_op, device_a5_op, AttentionOperateBase
from .prompt_flash_attn import PromptFlashAttention
from .fused_attn_score import FlashAttentionScore
from .ascend_laser_attention import AscendLaserAttention
from ...utils.get_platform import get_npu_device, NPUDevice
from ...utils.exception import ParametersInvalid
from ...utils.logs.logging import logger

TEST_COUNT = 5
WARM_UP_COUNT = 2
ATTN_DICT = {}


def attention_math(query, key, value, attn_mask, scale, head_first=False):
    if not head_first:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
    dtype = query.dtype
    device = query.device
    q_seqlen, kv_seqlen = query.size(-2), key.size(-2)
    query = query * scale
    attn = query @ key.transpose(-2, -1)
    attn = attn.to(dtype=torch.float32)
    if attn_mask is not None:
        if attn_mask.dim() not in [2, 4] or attn_mask.size(-2) != q_seqlen or attn_mask.size(-1) != kv_seqlen:
            raise ParametersInvalid("The attn_mask must be a 2D tensor with shape [q_seqlen, kv_seqlen],"
                                    " or a 4D tensor with shape [batch_size, num_heads, q_seqlen, kv_seqlen]")
        attn_bias = torch.zeros(q_seqlen, kv_seqlen, dtype=dtype).to(device)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        attn += attn_bias
    attn = torch.softmax(attn, dim=-1)
    attn = attn.to(dtype=dtype)
    output = attn @ value
    if not head_first:
        output = output.transpose(1, 2)
    return output


def get_attention_function_static(attn_param):
    logger.debug(f"Begin to get attention function in static mode. Current parameters are {attn_param.to_str()}")

    hash_key = attn_param.to_hash()
    if hash_key in ATTN_DICT:
        op_type, layout = ATTN_DICT[hash_key]
    else:
        logger.debug("Unable to locate the cached result in the static table; "
                     "defaulting to the standard operation type and layout.")

        op_type = "prompt_flash_attn"  # default
        layout = "BNSD"
    return get_attention_function(attn_param, op_type, layout)


def get_attention_function(attn_param, op_type, layout):
    npu_device = get_npu_device()
    if npu_device == NPUDevice.Duo:
        op_registry = device_duo_op.get_all()
    elif npu_device == NPUDevice.A2:
        op_registry = device_800_op.get_all()
    elif npu_device == NPUDevice.A5:
        op_registry = device_a5_op.get_all()
    else:
        raise ParametersInvalid("Platform invalid. Please check env.")

    if op_type not in op_registry:
        raise ParametersInvalid(f"The 'op_type':{op_type} is not supported. "
                                f"The list of supported options is {op_registry.keys()}")

    op = op_registry[op_type]

    if layout not in op.supported_layout:
        raise ParametersInvalid(f"The 'layout':{layout} is not supported. "
                                f"The list of supported options is {op.supported_layout}")

    if attn_param.dtype not in op.supported_dtype:
        raise ParametersInvalid(f"The input dtype:{attn_param.dtype} is not supported. "
                                f"The list of supported options is {op.supported_dtype}")

    func = getattr(op, op.layout_to_func[layout])

    return partial(func, attn_param)


@lru_cache_by_attn_param(maxsize=512)
def get_attention_function_runtime(attn_param, query, key, value, attn_mask=None, scale=None):
    logger.debug(f"Begin to get attention function in runtime mode. Current parameters are {attn_param.to_str()}")
    npu_device = get_npu_device()
    if npu_device == NPUDevice.Duo:
        all_op = device_duo_op.get_all()
    elif npu_device == NPUDevice.A2:
        all_op = device_800_op.get_all()
    elif npu_device == NPUDevice.A5:
        all_op = device_a5_op.get_all()
    else:
        raise ParametersInvalid("Platform invalid.")

    func_lists = get_test_func_lists(attn_param, all_op)
    if len(func_lists) == 0:
        logger.debug("The runtime function list is None.")
        return attention_math

    cost_time_lists = get_all_func_forward_time(func_lists, query, key, value, attn_param, attn_mask, scale)
    if len(cost_time_lists) == 0:
        logger.debug("The cost time list is None.")
        return attention_math

    func_list = min(cost_time_lists, key=lambda x: x[3])  # 3: cost time is in place 3
    logger.debug(f"Got the most time-efficient function. "
                 f"Op name: {func_list[0]}, layout: {func_list[1]}, cost time: {func_list[3] * 1000}ms")

    return partial(func_list[2], attn_param)


def get_test_func_lists(attn_param: AttentionParam, all_op):
    func_lists = []
    if not all_op:
        raise ParametersInvalid(f"all_op is none!")
    for name, op in all_op.items():
        if not op.is_supported_dtype(attn_param.dtype):
            logger.debug(
                f"The input data type[{attn_param.dtype}] is not in the range supported by op {name}.")
            continue
        if not op.is_supported_shape(attn_param):
            logger.debug(
                f"The input data shape is not in the range supported by op {name}.")
            continue
        for layout in op.supported_layout:
            func_lists.append([name, layout, getattr(op, op.layout_to_func[layout])])
    return func_lists


def get_all_func_forward_time(func_lists, query, key, value, attn_param, attn_mask=None, scale=None):
    cost_time_lists = []
    for func_list in func_lists:
        name, layout, func = func_list
        try:
            for _ in range(WARM_UP_COUNT):
                out = func(attn_param, query, key, value, attn_mask, scale)

            torch.npu.synchronize()
            begin = time.time()
            for _ in range(TEST_COUNT):
                out = func(attn_param, query, key, value, attn_mask, scale)
            torch.npu.synchronize()
            end = time.time()
            cost_time = (end - begin) / TEST_COUNT

            logger.debug(f"Op name: {name}, layout: {layout}, cost time: {cost_time * 1000}ms")
            cost_time_lists.append([name, layout, func, cost_time])
        except Exception as e:
            logger.debug(f"Op name: {name}, layout: {layout}, get exception {e}.")
    return cost_time_lists
