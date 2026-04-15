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
from ..utils import ParametersInvalid, file_utils


def check_input_params(x, cos, sin, rotated_mode, head_first, fused):
    if not isinstance(x, torch.Tensor):
        raise ParametersInvalid(f"The data type of input x must be torch.Tensor, but got {type(x)}.")
    if not isinstance(cos, torch.Tensor):
        raise ParametersInvalid(f"The data type of input cos must be torch.Tensor, but got {type(cos)}.")
    if not isinstance(sin, torch.Tensor):
        raise ParametersInvalid(f"The data type of input sin must be torch.Tensor, but got {type(sin)}.")
    if not isinstance(rotated_mode, str):
        raise ParametersInvalid(f"The data type of input rotated_mode must be str, but got {type(rotated_mode)}.")
    if not isinstance(head_first, bool):
        raise ParametersInvalid(f"The data type of input head_first must be bool, but got {type(head_first)}.")
    if not isinstance(fused, bool):
        raise ParametersInvalid(f"The data type of input fused must be bool, but got {type(fused)}.")
    if x.dim() != 4:    # 4: BNSD/BSND/SBND输入dim
        raise ParametersInvalid(f"The dimensional of input x must be a 4, but got {x.dim()}.")
    if cos.dim() not in [2, 4]:    # 2: SD输入dim; 4: 11SD/1S1D/S11D输入dim
        raise ParametersInvalid(f"The dimensional of input cos must be a 2 or 4, but got {cos.dim()}.")
    if sin.dim() not in [2, 4]:    # 2: SD输入dim; 4: 11SD/1S1D/S11D输入dim
        raise ParametersInvalid(f"The dimensional of input sin must be a 2 or 4, but got {sin.dim()}.")
    if cos.dim() != sin.dim():
        raise ParametersInvalid(f"The dimensional of input cos must be equal to the dimensional of input sin, "
                                f"but {cos.dim()} != {sin.dim()}.")


def reshape_for_broadcast(x, cos, sin, head_first=False):
    ndim = x.ndim
    if head_first:
        shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return cos.view(*shape), sin.view(*shape)


def rotary_position_embedding(x: torch.Tensor,
                              cos: torch.Tensor,
                              sin: torch.Tensor,
                              rotated_mode: str = "rotated_half",
                              head_first: bool = False,
                              fused: bool = True) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        x (torch.Tensor):
            Query or key tensor to apply rotary embeddings. x must be 4-dimensional.
            The supported layout: [B,N,S,D], [B,S,N,D], [S,B,N,D].
            Input x could be represented as [x_0, x_1, ... , x_d/2-1, x_d/2, x_d/2+1, ... , x_d-1].
        cos (torch.Tensor):
            Precomputed cos frequency tensor for complex exponentials. cos must be 2 or 4-dimensional.
            Correspongding to the input x, the supported layout: [S, D], [1,1,S,D], [1,S,1,D], [S,1,1,D].
        sin (torch.Tensor):
            Precomputed sin frequency tensor for complex exponentials. sin must be 2 or 4-dimensional.
            Correspongding to the input x, the supported layout: [S, D], [1,1,S,D], [1,S,1,D], [S,1,1,D].
        rotated_mode (str):
            If `rotated_half`: rotate x to [-x_d/2, -x_d/2+1, ... , -x_d-1, x_0, x_1, ... , x_d/2-1].
            If `rotated_interleaved`: rotate x to [-x_1, x_0, -x_3, x_2, ... , -x_d-1, x_d-2].
        head_first (bool):
            In the layout of x, if N is before S, set to True; otherwise, set to False.
        fused (bool): 
            If fused is True, using high-performance RoPE operator.

    Returns:
        (torch.Tensor): modified query or key tensor with rotary embeddings.
    """

    check_input_params(x, cos, sin, rotated_mode, head_first, fused)
    if cos.dim() == 2 and sin.dim() == 2:    # 2: SD输入dim
        cos, sin = reshape_for_broadcast(x, cos, sin, head_first=head_first)

    match rotated_mode:
        case "rotated_half":
            mode = "half"
        case "rotated_interleaved":
            mode = "interleave"
        case _:
            raise ParametersInvalid(f"Unsupported rotated_mode: {rotated_mode}. The supported "
                                    "rotated_mode must be 'rotated_half' or 'rotated_interleaved'")

    x_in = x.to(cos.dtype)

    if fused:
        x_out = torch_npu.npu_rotary_mul(x_in, cos, sin, mode)

    elif mode == "interleave":
        # Used for HunyuanDiT, OpenSora, Flux, CogVideox
        x_real, x_imag = x_in.reshape(*x_in.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        x_out = x_in * cos + x_rotated * sin
    else:
        # Used for OpenSoraPlan, Stable Audio
        x_real, x_imag = x_in.reshape(*x_in.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
        x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        x_out = x_in * cos + x_rotated * sin

    return x_out.type_as(x)