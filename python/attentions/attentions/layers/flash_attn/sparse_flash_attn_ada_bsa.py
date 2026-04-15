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

from .. import _custom_ops as ops


def get_estimate_mask(
    query, key, value,
    scale=None,
    head_num=None,
    is_causal=False,
    input_layout="BNSD",
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    keep_sink=True,
    keep_recent=True,
    sparsity=0.0,
    cdf_threshold=1.0,
    sparse_size=128,
    stride=8
):
    smask, sct = ops.sparse_block_estimate(
        query,
        key,
        actual_seq_lengths=None,
        actual_seq_lengths_kv=None,
        input_layout=input_layout,
        stride=stride,
        sparse_size=sparse_size,
        num_heads=head_num,
        num_key_value_heads=head_num,
        scale_value=scale / stride,
        threshold=cdf_threshold,
        causal=is_causal,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        row_sparse=1.0 - sparsity
    )
    return smask, sct


def ada_block_sparse_attention(
    query, key, value,
    smask, sct,
    scale=None,
    head_num=None,
    is_causal=False,
    input_layout="BNSD",
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    sparse_size=128
):
    out = ops.block_sparse_attention(
        query,
        key,
        value,
        num_heads=head_num,
        num_key_value_heads=head_num,
        input_layout=input_layout,
        scale_value=scale,
        causal=is_causal,
        sparse_size=sparse_size,
        sparse_mask=smask,
        sparse_count_table=sct
    )
    return out