#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# 
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Tuple, List, Optional
import math
import torch
from . import register_ops
from ..utils import ParametersInvalid


def laser_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor | None = None,
    alibi_mask: torch.Tensor | None = None,
    drop_mask: torch.Tensor | None = None,
    scale_value: float | torch.Tensor = 1.0,
    head_num: int = 2,
    input_layout: str = "BNSD",
    keep_prob: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 1,
    is_high_precision: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return getattr(torch.ops.attentions, "la")(
        query=query,
        key=key,
        value=value,
        atten_mask=atten_mask,
        alibi_mask=alibi_mask,
        drop_mask=drop_mask,
        scale_value=scale_value,
        head_num=head_num,
        input_layout=input_layout,
        keep_prob=keep_prob,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        is_highPrecision=is_high_precision,
    )


@register_ops.register_attentions_fake_op("la")
def laser_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor = None,
    alibi_mask: torch.Tensor = None,
    drop_mask: torch.Tensor = None,
    scale_value: float = 1.0,
    head_num: int = 2,
    input_layout: str = "BNSD",
    keep_prob: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 1,
    is_high_precision: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    softmax_log_max_sum = torch.empty(
        [query.shape[0], query.shape[1], query.shape[2]],
        device=query.device, dtype=query.dtype
    )
    output = torch.empty_like(query)
    return softmax_log_max_sum, output

def rain_fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    select_idx: torch.Tensor,
    select_num_idx: torch.Tensor,
    blockshape: List[int],
    attn_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[List[int]] = None,
    actual_seq_kvlen: Optional[List[int]] = None,
    block_table: Optional[torch.Tensor] = None,
    q_input_layout: str = 'TND',
    kv_input_layout: str = 'TND',
    head_num: int = 1,
    mask_type: int = 0,
    scale: float = 1.0,
    inner_precise: int = 1,
    block_size: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    return getattr(torch.ops.attentions, "rainfusionattention")(
        query=query,
        key=key,
        value=value,
        select_idx=select_idx,
        select_num_idx=select_num_idx,
        blockshape=blockshape,
        attn_mask=attn_mask,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        block_table=block_table,
        q_input_layout=q_input_layout,
        kv_input_layout=kv_input_layout,
        head_num=head_num,
        mask_type=mask_type,
        scale=scale,
        inner_precise=inner_precise,
        block_size=block_size
    )


@register_ops.register_attentions_fake_op("rainfusionattention")
def rain_fusion_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    select_idx: torch.Tensor,
    select_num_idx: torch.Tensor,
    blockshape: List[int],
    attn_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[List[int]] = None,
    actual_seq_kvlen: Optional[List[int]] = None,
    block_table: Optional[torch.Tensor] = None,
    q_input_layout: str = 'TND',
    kv_input_layout: str = 'TND',
    head_num: int = 1,
    mask_type: int = 0,
    scale: float = 1.0,
    inner_precise: int = 1,
    block_size: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    softmax_lse = torch.empty(
        [query.shape[0], query.shape[1], query.shape[2]],
        device=query.device, dtype=query.dtype
    )
    output = torch.empty_like(query)
    return output, softmax_lse


def sparse_block_estimate(
    query: torch.Tensor,
    key: torch.Tensor,
    actual_seq_lengths: Optional[List[int]] = None,
    actual_seq_lengths_kv: Optional[List[int]] = None,
    input_layout: str = 'BNSD',
    stride: int = 8,
    sparse_size: int = 128,
    num_heads: int = 1,
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    threshold: float = 1.0,
    causal: bool = False,
    keep_sink: bool = True,
    keep_recent: bool = True,
    row_sparse: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    return getattr(torch.ops.attentions, "sparse_block_estimate")(
        query=query,
        key=key,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        input_layout=input_layout,
        stride=stride,
        sparse_size=sparse_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale_value,
        threshold=threshold,
        causal=causal,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        row_sparse=row_sparse
    )


@register_ops.register_attentions_fake_op("sparse_block_estimate")
def sparse_block_estimate_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    actual_seq_lengths: Optional[List[int]] = None,
    actual_seq_lengths_kv: Optional[List[int]] = None,
    input_layout: str = 'BNSD',
    stride: int = 8,
    sparse_size: int = 128,
    num_heads: int = 1,
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    threshold: float = 1.0,
    causal: bool = False,
    keep_sink: bool = True,
    keep_recent: bool = True,
    row_sparse: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, nq, s, d = 0, 0, 0, 0
    if input_layout == "BNSD":
        b, nq, s, d = query.shape
    elif input_layout == "BSND":
        b, s, nq, d = query.shape
    else:
        raise ParametersInvalid(f"The input_layout only support 'BNSD' and 'BSND' now, but got {input_layout}")
    seqlen_sparse = int((s + sparse_size - 1) / sparse_size)
    seqlen_sparse_align32 = (seqlen_sparse + 31) / 32 * 32
    sparse_mask_shape = (b, nq, seqlen_sparse, seqlen_sparse_align32)
    sparse_count_table_shape = (b, nq, seqlen_sparse)

    sparse_mask = torch.empty(
        sparse_mask_shape,
        device=query.device, dtype=torch.int8
    )
    sparse_count_table = torch.empty(
        sparse_count_table_shape,
        device=query.device, dtype=torch.int32
    )
    return sparse_mask, sparse_count_table


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_mask: torch.Tensor,
    sparse_count_table: torch.Tensor,
    input_layout: str = 'BNSD',
    sparse_size: int = 128,
    num_heads: int = 1,
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    causal: bool = False,
    inner_precise: int = 1,
    pre_tokens: int = 214748647,
    next_tokens: int = 214748647,
    actual_seq_lengths: Optional[List[int]] = None,
    actual_seq_lengths_kv: Optional[List[int]] = None,
) -> torch.Tensor:
    return getattr(torch.ops.attentions, "block_sparse_attention")(
        query=query,
        key=key,
        value=value,
        sparse_mask=sparse_mask,
        sparse_count_table=sparse_count_table,
        input_layout=input_layout,
        sparse_size=sparse_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale_value,
        causal=causal,
        inner_precise=inner_precise,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv
    )


@register_ops.register_attentions_fake_op("block_sparse_attention")
def block_sparse_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_mask: torch.Tensor,
    sparse_count_table: torch.Tensor,
    input_layout: str = 'BNSD',
    sparse_size: int = 128,
    num_heads: int = 1,
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    causal: bool = False,
    inner_precise: int = 1,
    pre_tokens: int = 214748647,
    next_tokens: int = 214748647,
    actual_seq_lengths: Optional[List[int]] = None,
    actual_seq_lengths_kv: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    return output
