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

import math
import torch_npu
import torch.nn.functional as F
from ...utils import ParametersInvalid


def fa_block_quant_preprocess(input_tensor, block_size=128, dst_type=torch_npu.float8_e4m3fn, **kwargs):
    """
    Preprocess for FA quant. Input layout must be 'BNSD' or 'BSND'.
    Args:
        input_tensor (torch.Tensor): Input tensor to be quantized.
        block_size (int, optional): Block size for quantization. Support 128/256. Default: 128.
        dst_type (torch.dtype, optional): Target quantization data type. Default: torch_npu.float8_e4m3fn.
        **kwargs: 
            layout (str): Tensor layout format, supports 'BNSD' (Batch, Num_heads, Seq_len, Dim) 
                         or 'BSND' (Batch, Seq_len, Num_heads, Dim).

    Returns:
        torch.Tensor: Preprocessed tensor ready for FA block quantization.
    """

    if len(input_tensor.shape) != 4:
        raise ParametersInvalid(f"fa block quant preprocess only support qkv quant, dim = 4, \
                                but got {len(input_tensor.shape)}.")

    layout = kwargs.get("layout", "BNSD")
    if layout == "BNSD":
        b, n, s, d = input_tensor.shape
    elif layout == "BSND":
        input_tensor = input_tensor.transpose(1, 2)
        b, n, s, d = input_tensor.shape
    else:
        raise ValueError("unsupport layout")

    # Padding is automatically applied to meet block alignment requirements.
    if not s % block_size == 0:
        padding_length = (block_size - (s % block_size)) % block_size
        input_tensor = F.pad(input_tensor, (0, 0, 0, padding_length))

    input_tensor = input_tensor.reshape(b, n, math.ceil(s / block_size), -1)
    input_quant, input_scale = torch_npu.npu_dynamic_quant(input_tensor, dst_type=dst_type)

    if layout == "BNSD":
        input_quant = input_quant.reshape(b, n, -1, d)[:, :, :s, :]
    elif layout == "BSND":
        input_quant = input_quant.transpose(1, 2).reshape(b, -1, n, d)[:, :s, :, :]

    return input_quant, input_scale.unsqueeze(-1)
