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
import torch_npu
from .attention_operate import AttentionOperateBase, register_op_800
from ...utils.exception import ParametersInvalid
from ...utils import file_utils
from .. import _custom_ops as ops
from .common import AttentionParam

SEQ_LEN_PAD_BASE = 256
MAX_TOKEN = 2**31 - 1
MAX_DIM = 128
MIN_SEQLEN_SELF = 4000
MIN_SEQLEN_CROSS = 118404
MAX_SEQLEN_CROSS = 119056
SEQLEN_BASE = 256
DIM_BASE = 128
SEQLEN_INDEX = -2
DIM_INDEX = -1



@register_op_800("ascend_laser_attention")
class AscendLaserAttention(AttentionOperateBase):
    supported_layout = ["BNSD"]
    supported_dtype = [torch.float16, torch.bfloat16]

    @staticmethod
    def pad(input_tensor, base=256, dim=-2):
        shape_value = input_tensor.size(dim)
        if shape_value % base != 0:
            pad_size = ((shape_value // base) + 1) * base - shape_value
            padding_shape = list(input_tensor.shape)
            padding_shape[dim] = pad_size
            padding = torch.zeros(padding_shape, dtype=input_tensor.dtype, device=input_tensor.device)
            return torch.cat([input_tensor, padding], dim=dim)
        return input_tensor

    @staticmethod
    def la_preprocess_input(query, key, value):
        if query.dtype != torch.float16:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)

        query = AscendLaserAttention.pad(query, SEQLEN_BASE, SEQLEN_INDEX)
        query = AscendLaserAttention.pad(query, DIM_BASE, DIM_INDEX)

        key = AscendLaserAttention.pad(key, SEQLEN_BASE, SEQLEN_INDEX)
        key = AscendLaserAttention.pad(key, DIM_BASE, DIM_INDEX)

        value = AscendLaserAttention.pad(value, SEQLEN_BASE, SEQLEN_INDEX)
        value = AscendLaserAttention.pad(value, DIM_BASE, DIM_INDEX)
        return query, key, value

    @staticmethod
    def la_postprocess_output(attention_out, dtype, qseqlen, head_dim):
        if dtype != attention_out.dtype:
            attention_out = attention_out.to(torch.float16).to(dtype)
        attention_out = attention_out[:, :, :qseqlen, :head_dim]
        return attention_out

    @classmethod
    def is_supported_shape(cls, attn_param: AttentionParam) -> bool:
        if attn_param.head_dim > MAX_DIM:
            return False
        if attn_param.q_seqlen == attn_param.kv_seqlen:
            return attn_param.q_seqlen >= MIN_SEQLEN_SELF
        else:
            return (MIN_SEQLEN_CROSS <= attn_param.q_seqlen <= MAX_SEQLEN_CROSS) and \
                (MIN_SEQLEN_CROSS <= attn_param.kv_seqlen <= MAX_SEQLEN_CROSS)

    @classmethod
    def forward_attn_bnsd(
            cls,
            attn_param: AttentionParam,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None,
            scale: torch.Tensor = None
    ) -> torch.Tensor:
        head_first = attn_param.head_first
        if not head_first:
            # input layout is bsnd
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        if mask is not None:
            mask = ~mask.to(torch.bool)

        new_query, new_key, new_value = AscendLaserAttention.la_preprocess_input(query, key, value)
        pre_tokens = MAX_TOKEN
        if attn_param.kv_seqlen % SEQ_LEN_PAD_BASE != 0:
            pre_tokens = (attn_param.kv_seqlen // SEQ_LEN_PAD_BASE + 1) * SEQ_LEN_PAD_BASE - attn_param.kv_seqlen

        _, output1 = ops.laser_attention(
            new_query, new_key, new_value, None, None, None,
            scale, attn_param.head_num, "BNSD", 1.0, pre_tokens, 1, True
        )
        out = AscendLaserAttention.la_postprocess_output(output1, query.dtype, attn_param.q_seqlen, attn_param.head_dim)

        if not head_first:
            out = out.transpose(1, 2)
        return out
