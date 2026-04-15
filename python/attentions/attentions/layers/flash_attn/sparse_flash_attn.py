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


from typing import Optional
import torch
import torch_npu
from .sparse_flash_attn_rf_v2 import (
    rain_fusion_attention,
    get_blockwise_mask,
    do_tensor_inv_rearrange,
    do_tensor_rearrange_pooling
)
from .sparse_flash_attn_ada_bsa import get_estimate_mask, ada_block_sparse_attention
from ...utils.exception import ParametersInvalid
MAX_TOKEN = 2147483647


def check_params(input_layout, sparse_type):
    if input_layout not in ['BSND', 'BNSD']:
        raise ParametersInvalid(f"The input_layout must in ['BSND', 'BNSD'], but got {input_layout}.")
    if sparse_type not in [None, 'rf_v2', 'ada_bsa']:
        raise ParametersInvalid(f"sparse_type must be None, 'rf_v2' or 'ada_bsa', but got {sparse_type}.")


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: Optional[bool] = False,
    head_num: int = 1,
    input_layout: str = "BNSD",
    inner_precise: int = 0,
    # 稀疏参数
    sparse_type: Optional[str] = None,
    txt_len: int = 0,
    block_size: int = 128,
    latent_shape_q: Optional[list] = None,
    latent_shape_k: Optional[list] = None,
    keep_sink: Optional[bool] = True,
    keep_recent: Optional[bool] = True,
    cdf_threshold: float = 1.0,
    sparsity: float = 0.0,
    **kwargs
):
    """
    Args:
        q ('torch.Tensor'):
            If 'input_layout' is 'BNSD', the shape of q is [batch, head, qseqlen, headdim];
            If 'input_layout' is 'BSND', the shape of q is [batch, qseqlen, head, headdim];
        k ('torch.Tensor'):
            If 'input_layout' is 'BNSD', the shape of k is [batch, head, kseqlen, headdim];
            If 'input_layout' is 'BSND', the shape of k is [batch, kseqlen, head, headdim];
        v ('torch.Tensor'):
            If 'input_layout' is 'BNSD', the shape of v is [batch, head, vseqlen, headdim];
            If 'input_layout' is 'BSND', the shape of v is [batch, vseqlen, head, headdim];
        attn_mask ('torch.Tensor', *optional*, defaults to `None`):
            Reserved for future use (planned for regularization).
        scale ('float', *optional*, defaults to `None`):
            The input scale of attention calculation formula, if not provided, will be set to 'headdim ** -0.5'.
        is_causal (bool, default to False):
            Whether to apply causal mask to the attention matrix.
        head_num (int, default to 1):
            The head of qkv.
        input_layout (str, default to 'BSND'):
            The tensor layout, either 'BSND' or 'BNSD'.
        inner_precise (int, default to 0):
            0 represents high-precision; 1 represents high-performance.
        sparse_type (str, default to None):
            Sparse type, only supports: 'rf_v2', 'ada_bsa'.
        txt_len:
            Length of text sequence. Only takes effect when sparse_type is 'rf_v2'.
        block_size (int, default to 128):
            Only supports 128.
        latent_shape_q (list, default to None):
            (t, h, w), t**h*w = qseqlen. Only takes effect when sparse_type is 'rf_v2'.
        latent_shape_k (list, default to None):
            (t, h, w), t**h*w = kseqlen. Only takes effect when sparse_type is 'rf_v2'.
        keep_sink (bool, default to True):
            Only takes effect when sparse_type ims 'ada_bsa'.
        keep_recent (bool, default to True):
            Only takes effect when sparse_type is 'ada_bsa'.
        cdf_threshold (float, default to 1.0):
            Only takes effect when sparse_type is 'ada_bsa'.
        sparsity:
            Sparse ratio, the value range is [0, 1], where 0 represents not using sparse algo.
    """
    check_params(input_layout, sparse_type)
    batch, head_dim = q.shape[0], q.shape[-1]
    scale = head_dim ** -0.5 if scale is None else scale

    if sparse_type == "rf_v2":
        q, k, v, qkv_pool = do_tensor_rearrange_pooling(
            q, k, v, txt_len, block_size, latent_shape_q, latent_shape_k, input_layout
        )
        select_idx, select_num_idx = get_blockwise_mask(
                qkv_pool, txt_len, sparsity, scale, block_size, latent_shape_q, latent_shape_k, input_layout)

        if input_layout == "BSND":
            q_seq, kv_seq = q.shape[1], k.shape[1]
            layout = "TND"
            q = q.reshape(-1, head_num, head_dim)
            k = k.reshape(-1, head_num, head_dim)
            v = v.reshape(-1, head_num, head_dim)
        else:
            q_seq, kv_seq = q.shape[2], k.shape[2]
            layout = input_layout
        actual_seq_lengths = [q_seq for _ in range(batch)]
        actual_seq_lengths_kv = [kv_seq for _ in range(batch)]

        out = rain_fusion_attention(
            q, k, v,
            scale=scale,
            head_num=head_num,
            input_layout=layout,
            select_idx=select_idx,
            select_num_idx=select_num_idx,
            blockshape=[block_size, block_size],
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            inner_precise=inner_precise
        )
        if layout == "TND":
            out = out.reshape(batch, q_seq, head_num, head_dim)
        out = do_tensor_inv_rearrange(out, txt_len, latent_shape_q, latent_shape_k, input_layout)
    elif sparse_type == "ada_bsa":
        smask, sct = get_estimate_mask(
            q, k, v,
            scale=scale,
            head_num=head_num,
            is_causal=is_causal,
            input_layout=input_layout,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            sparsity=sparsity,
            cdf_threshold=cdf_threshold,
            sparse_size=block_size
        )
        out = ada_block_sparse_attention(
            q, k, v,
            smask, sct,
            scale=scale,
            head_num=head_num,
            is_causal=is_causal,
            input_layout=input_layout,
            sparse_size=block_size
        )
    elif sparse_type is None:
        out = torch_npu.npu_fusion_attention(
            q, k, v,
            input_layout=input_layout,
            scale=scale,
            pre_tockens=MAX_TOKEN,
            next_tockens=MAX_TOKEN,
            head_num=head_num)[0]
    return out
