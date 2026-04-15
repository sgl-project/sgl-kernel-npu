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

import torch
from ..passes.register_pattern_to_pass import PatternBase

if hasattr(torch.npu, "is_available"):
    npu_available = torch.npu.is_available()
if npu_available:
    import torch_npu
    import mindiesd


def create(dtype, epsilon=1e-6):
    class AdaLayerNormZeroPatternDiffusers(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"
        
        @staticmethod
        def inputs():
            x = torch.empty(2, 2, 2, dtype=dtype, device="meta")
            scale = torch.empty(2, 2, dtype=dtype, device="meta")
            shift = torch.empty(2, 2, dtype=dtype, device="meta")
            return [x, scale, shift]

        @staticmethod
        def pattern(x, scale, shift):
            # Reference:
            # github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/normalization.py#L131
            
            def func(x, scale, shift):
                ln_out = torch.nn.LayerNorm(
                    x.shape[-1], 
                    elementwise_affine=False,
                    eps=epsilon, 
                    dtype=x.dtype, 
                    device=x.device)(x)

                out = ln_out * (1 + scale[:, None]) + shift[:, None]
                return out

            return func(x, scale, shift)

        @staticmethod
        def replacement(x, scale, shift):
            norm = torch.nn.LayerNorm(x.shape[-1], eps=epsilon, dtype=x.dtype, device=x.device)
            
            def func(x, scale, shift):
                return mindiesd.layernorm_scale_shift(
                    layernorm=norm,
                    x=x,
                    scale=scale[:, None],
                    shift=shift[:, None],
                    fused=True
                )
            return func(x, scale, shift)

    return AdaLayerNormZeroPatternDiffusers

AdaLayerNormPatternGroup = [create(dtype=torch.bfloat16, epsilon=1e-6)]