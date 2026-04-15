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
import math
import torch
from einops import rearrange
from .. import _custom_ops as ops
from . import attention_forward
from ...utils.exception import ParametersInvalid
from ...utils import file_utils


def avgpool(input_tensor, pool_size=128, input_layout='BNSD'): # BSND in,  BSND out
    if input_layout == "BSND":
        batch, seqlen, headnum, dim = input_tensor.shape
        
        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size
        
        if num_full_blocks > 0:
            full_blocks = input_tensor[:, :num_full_blocks * pool_size, :, :]
            full_blocks_reshaped = full_blocks.view(batch, num_full_blocks, pool_size, headnum, dim)
            full_pooled = full_blocks_reshaped.mean(dim=2) 
        else:
            full_pooled = torch.empty(0, device=input_tensor.device)
        if tail_size > 0:
            tail_block = input_tensor[:, num_full_blocks * pool_size:, :, :]
            tail_reshaped = tail_block.view(batch, 1, tail_size, headnum, dim)
            tail_pooled = tail_reshaped.mean(dim=2)
        else:
            tail_pooled = torch.empty(0, device=input_tensor.device)

        if num_full_blocks > 0 and tail_size > 0:
            output_tensor = torch.cat([full_pooled, tail_pooled], dim=1)
        elif num_full_blocks > 0:
            output_tensor = full_pooled
        else:
            output_tensor = tail_pooled
    elif input_layout == "BNSD":
        batch, headnum, seqlen, dim = input_tensor.shape
        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size
        if num_full_blocks > 0:
            full_blocks = input_tensor[:, :, :num_full_blocks * pool_size, :]
            full_blocks_reshaped = full_blocks.view(batch, headnum, num_full_blocks, pool_size, dim)
            full_pooled = full_blocks_reshaped.mean(dim=3)
        else:
            full_pooled = torch.empty(0, device=input_tensor.device)
        if tail_size > 0:
            tail_block = input_tensor[:, :, num_full_blocks * pool_size:, :]
            tail_reshaped = tail_block.view(batch, headnum, 1, tail_size, dim)
            tail_pooled = tail_reshaped.mean(dim=3)
        else:
            tail_pooled = torch.empty(0, device=input_tensor.device)
        
        if num_full_blocks > 0 and tail_size > 0:
            output_tensor = torch.cat([full_pooled, tail_pooled], dim=2)
        elif num_full_blocks > 0:
            output_tensor = full_pooled
        else:
            output_tensor = tail_pooled
    return output_tensor


def get_mask_index(mask):
    b, n, s, _ = mask.shape
    device = mask.device
    
    mask_reshaped = mask.reshape(-1, s, s)
    batch_size = mask_reshaped.shape[0]
    
    row_indices = torch.arange(s, device=device).expand(batch_size, s, -1)
    sorted_vals = torch.where(mask_reshaped, row_indices, 1e9).to(torch.float32)
    sorted_vals, _ = torch.sort(sorted_vals, dim=-1)
    valid_count = mask_reshaped.sum(dim=-1, keepdim=True)
    keep_mask = row_indices < valid_count
    result = torch.where(keep_mask, sorted_vals, -1)
    
    pos_matrix = result.reshape(b, n, s, s).to(torch.int64)
    return pos_matrix


def get_blockwise_mask(
    qkv_pool,
    txt_len, sparsity, scale, pool_size,
    latent_shape_q, latent_shape_k,
    input_layout):
    tq, hq, wq = latent_shape_q
    first_frame_len = hq * wq

    query_pool, key_pool, value_pool = torch.chunk(qkv_pool, 3, dim=0)
    if input_layout == "BSND":
        attn_scores_head = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
    elif input_layout == "BNSD":
        attn_scores_head = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale
    score_matrix = torch.nn.functional.softmax(attn_scores_head, dim=-1)

    cols = score_matrix.shape[-1]
    
    keep_len = math.ceil(cols * (1 - sparsity))
    topk_values, _ = torch.topk(score_matrix, k=keep_len, dim=-1)
    thresholds = topk_values[..., -1:]
    mask = score_matrix >= thresholds
    text_block_num = (txt_len + pool_size - 1) // pool_size

    if text_block_num > 0:
        mask[:, :, -text_block_num:, :] = True
        mask[:, :, :, -text_block_num:] = True

    firstframe_block_num = (first_frame_len + pool_size - 1) // pool_size
    if firstframe_block_num > 0:
        mask[:, :, :firstframe_block_num, :] = True
        mask[:, :, :, :firstframe_block_num] = True
    select_idx = get_mask_index(mask)
    select_idx = select_idx[0].transpose(0, 1)
    select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
    return select_idx, select_num_idx


def rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout):
    '''
    b (f hn hb wn wb) n d -> b (f hn wn hb wb) n d
    or
    b n (f hn hb wn wb) d -> b n (f hn wn hb wb) d
    '''
    tq, hq, wq = latent_shape_q
    first_frame_len, frame_num = hq * wq, tq
    if input_layout == "BSND":
        b, s, n, d = tensor.shape

        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_first = tensor[:, :first_frame_len, :, :]
            tensor = tensor[:, first_frame_len:, :, :]
            tensor_hwt = rearrange(tensor, 'b (f h w) n d -> b f h w n d', f=frame_num - 1, h=hq, w=wq)
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=2)
                tensor_h_r = tensor_h_r.reshape(b, frame_num - 1, -1, n, d)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=3)
                tensor_w_r = tensor_w_r.reshape(b, frame_num - 1, -1, n, d)
            tensor_hwt = rearrange(tensor_hwt, 'b f (hn hb) (wn wb) n d -> b f (hn wn hb wb) n d', f=frame_num - 1,
                                   hb=8, wb=8, hn=hq // 8, wn=wq // 8)
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=2)
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=2)
            tensor_hwt = tensor_hwt.reshape(b, -1, n, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=1)
        else:
            tensor_hwt = rearrange(tensor, 'b (f hn hb wn wb) n d -> b (f hn wn hb wb) n d', f=frame_num, hb=8, wb=8,
                                hn=hq // 8, wn=wq // 8)
    elif input_layout == "BNSD":
        b, n, s, d = tensor.shape
        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_first = tensor[:, :, :first_frame_len, :]
            tensor = tensor[:, :, first_frame_len:, :]
            tensor_hwt = rearrange(tensor, 'b n (f h w) d -> b n f h w d', f=frame_num - 1, h=hq, w=wq)
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=3)
                tensor_h_r = tensor_h_r.reshape(b, n, frame_num - 1, -1, d)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=4)
                tensor_w_r = tensor_w_r.reshape(b, n, frame_num - 1, -1, d)
            tensor_hwt = rearrange(tensor_hwt, 'b n f (hn hb) (wn wb) d -> b n f (hn wn hb wb) d', f=frame_num - 1,
                                hb=8, wb=8, hn=hq // 8, wn=wq // 8)
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=3)
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=3)
            tensor_hwt = tensor_hwt.reshape(b, n, -1, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=2)
        else:
            tensor_hwt = rearrange(tensor, 'b n (f hn hb wn wb) d -> b n (f hn wn hb wb) d', f=frame_num, hb=8, wb=8,
                                hn=hq // 8, wn=wq // 8)

    return tensor_hwt


def inv_rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout): # BSND in ,  BSND out
    tq, hq, wq = latent_shape_q
    first_frame_len, frame_num = hq * wq, tq
    if input_layout == "BSND":
        b, s, n, d = tensor.shape

        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_first = tensor[:, :first_frame_len, :, :]
            tensor = tensor[:, first_frame_len:, :, :]
            tensor_hwt = rearrange(tensor, 'b (f h w) n d -> b f h w n d', f=frame_num - 1, h=hq, w=wq)
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=2)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=3)
            tensor_hwt = tensor_hwt.reshape(b, frame_num - 1, -1, n, d)
            tensor_hwt = rearrange(tensor_hwt, 'b f (hn wn hb wb) n d -> b f (hn hb) (wn wb) n d', f=frame_num - 1,
                                   hb=8, wb=8, hn=hq // 8, wn=wq // 8)
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=3)
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=2)
            tensor_hwt = tensor_hwt.reshape(b, -1, n, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=1)
        else:
            tensor_hwt = rearrange(tensor, 'b (f hn wn hb wb) n h -> b (f hn hb wn wb) n h', f=frame_num, hb=8, wb=8,
                                hn=hq // 8, wn=wq // 8)
    elif input_layout == "BNSD":
        b, n, s, d = tensor.shape
        if (hq % 8 != 0) or (wq % 8 != 0):
            tensor_first = tensor[:, :, :first_frame_len, :]
            tensor = tensor[:, :, first_frame_len:, :]
            tensor_hwt = rearrange(tensor, 'b n (f h w) d -> b n f h w d', f=frame_num - 1, h=hq, w=wq)
            if hq % 8 != 0:
                tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=3)
            if wq % 8 != 0:
                tensor_hwt, tensor_w_r = torch.split(tensor_hwt, wq - (wq % 8), dim=4)
            tensor_hwt = tensor_hwt.reshape(b, n, frame_num - 1, -1, d)
            tensor_hwt = rearrange(tensor_hwt, 'b n f (hn wn hb wb) d -> b n f (hn hb) (wn wb) d', f=frame_num - 1,
                                   hb=8, wb=8, hn=hq // 8, wn=wq // 8)
            if hq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_h_r), dim=3)
            if wq % 8 != 0:
                tensor_hwt = torch.cat((tensor_hwt, tensor_w_r), dim=4)
            tensor_hwt = tensor_hwt.reshape(b, n, -1, d)
            tensor_hwt = torch.cat([tensor_first, tensor_hwt], dim=2)
        else:
            tensor_hwt = rearrange(tensor, 'b n (f hn wn hb wb) h -> b n (f hn hb wn wb) h', f=frame_num, hb=8, wb=8,
                                hn=hq // 8, wn=wq // 8)
    return tensor_hwt

                
def do_tensor_rearrange_pooling(query, key, value, text_len, pool_size, latent_shape_q, latent_shape_k, input_layout):
    '''
    张量的分块重排 + 池化操作
    '''
    tensor = torch.cat((query, key, value), dim=0)
    if text_len != 0:
        if input_layout == "BSND":
            tensor_t = tensor[:, :text_len, :, :]
            tensor_i = tensor[:, text_len:, :, :]
        elif input_layout == "BNSD":
            tensor_t = tensor[:, :, :text_len, :]
            tensor_i = tensor[:, :, text_len:, :]
        tensor_i_2 = rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
        tensor_i_pool = avgpool(tensor_i_2, pool_size, input_layout)
        tensor_t_pool = avgpool(tensor_t, pool_size, input_layout)
        if input_layout == "BSND":
            tensor = torch.concat((tensor_i_2, tensor_t), dim=1)
            tensor_pool = torch.concat((tensor_i_pool, tensor_t_pool), dim=1)
        elif input_layout == "BNSD":
            tensor = torch.concat((tensor_i_2, tensor_t), dim=2)
            tensor_pool = torch.concat((tensor_i_pool, tensor_t_pool), dim=2)
    else:
        tensor = rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout)
        tensor_pool = avgpool(tensor, pool_size, input_layout)
    query_, key_, value_ = torch.chunk(tensor, 3, dim=0)
    return query_, key_, value_, tensor_pool
   


def do_tensor_inv_rearrange(tensor, text_len, latent_shape_q, latent_shape_k, input_layout):
    if text_len != 0:
        if input_layout == "BSND":
            tensor_t = tensor[:, -text_len:, :, :]
            tensor_i = tensor[:, :-text_len, :, :]

            tensor_i = inv_rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
            tensor = torch.concat((tensor_t, tensor_i), dim=1)
        elif input_layout == "BNSD":
            tensor_t = tensor[:, :, -text_len:, :]
            tensor_i = tensor[:, :, :-text_len, :]
            tensor_i = inv_rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
            tensor = torch.concat((tensor_t, tensor_i), dim=2)
    else:
        tensor = inv_rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout)

    return tensor


def do_tensor_pooling(tensor, text_len):
    tensor_t = tensor[:, :text_len, :, :]
    tensor_i = tensor[:, text_len:, :, :]

    tensor_i_pool = avgpool(tensor_i, pool_size=128)
    tensor_t_pool = avgpool(tensor_t, pool_size=128)

    tensor_pool = torch.concat((tensor_t_pool, tensor_i_pool), dim=1)
    return tensor_pool

  
def rain_fusion_attention(
    query,
    key,
    value,
    scale=None,
    head_num=None,
    input_layout="TND",
    select_idx=None,
    select_num_idx=None,
    blockshape=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    inner_precise=0
):
    
    out, _ = ops.rain_fusion_attention(
        query, key, value,
        select_idx, select_num_idx,
        blockshape,
        attn_mask=None,
        actual_seq_qlen=actual_seq_lengths,
        actual_seq_kvlen=actual_seq_lengths_kv,
        block_table=None,
        q_input_layout=input_layout,
        kv_input_layout=input_layout,
        head_num=head_num,
        mask_type=0, scale=scale,
        inner_precise=inner_precise,
        block_size=0)
    
    return out