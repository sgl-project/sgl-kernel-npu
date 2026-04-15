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

import torch
import torch_npu
from .attention_operate import AttentionOperateBase, register_op_duo, register_op_800, register_op_a5
from .common import AttentionParam


@register_op_duo("prompt_flash_attn")
@register_op_800("prompt_flash_attn")
@register_op_a5("prompt_flash_attn")
class PromptFlashAttention(AttentionOperateBase):
    supported_layout = ["BNSD", "BSND", "BSH"]
    supported_dtype = [torch.float16, torch.bfloat16, torch.int8]
    
    @classmethod
    def is_supported_shape(cls, attn_param: AttentionParam) -> bool:
        return True
    
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
        out = torch_npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            atten_mask=mask,
            input_layout="BNSD",
            scale_value=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            num_heads=attn_param.head_num)
        if not head_first:
            out = out.transpose(1, 2)
        return out

    @classmethod
    def forward_attn_bsnd(
        cls,
        attn_param: AttentionParam,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        scale: torch.Tensor = None
    ) -> torch.Tensor:
        head_first = attn_param.head_first
        if head_first:
            # input layout is bnsd
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        if mask is not None:
            mask = ~mask.to(torch.bool)
        out = torch_npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            atten_mask=mask,
            input_layout="BSND",
            scale_value=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            num_heads=attn_param.head_num)
        if head_first:
            out = out.transpose(1, 2)
        return out

    @classmethod
    def forward_attn_bsh(
        cls,
        attn_param: AttentionParam,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        scale: torch.Tensor = None
    ) -> torch.Tensor:
        head_first = attn_param.head_first
        if head_first:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        # input layout is bsnd
        query = query.reshape(attn_param.batch_size, attn_param.q_seqlen, attn_param.head_num * attn_param.head_dim)
        key = key.reshape(attn_param.batch_size, attn_param.kv_seqlen, attn_param.head_num * attn_param.head_dim)
        value = value.reshape(attn_param.batch_size, attn_param.kv_seqlen, attn_param.head_num * attn_param.head_dim)
        if mask is not None:
            mask = ~mask.to(torch.bool)
        out = torch_npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            atten_mask=mask,
            input_layout="BSH",
            scale_value=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            num_heads=attn_param.head_num)
        if not head_first:
            out = out.reshape(attn_param.batch_size, attn_param.q_seqlen, attn_param.head_num, attn_param.head_dim)
        else:
            out = out.reshape(attn_param.batch_size, attn_param.q_seqlen, attn_param.head_num, attn_param.head_dim)
            out = out.transpose(1, 2)        
        return out