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
    if "2.6.0" in torch.__version__:
        _dtype_cast_func = torch.ops.npu.npu_dtype_cast.default
        _eps_in_bf16 = torch.tensor(epsilon, dtype=torch.bfloat16, device="cpu").item()
    else:
        _dtype_cast_func = torch.ops.npu._npu_dtype_cast.default
        _eps_in_fp32 = torch.tensor(epsilon, dtype=torch.float32, device="cpu").item()


    class RMSNormPattern(PatternBase):
        @staticmethod
        def name():
            return __class__.__name__ + f"-{dtype}"

        @staticmethod
        def inputs():
            hidden_states = torch.empty(2, 2, 2, 2, dtype=dtype, device="meta")
            weight = torch.empty(2, dtype=dtype, device="meta")
            return [hidden_states, weight]
        
        @staticmethod
        def pattern(hidden_states, weight):
            def func(hidden_states, weight):
                '''
                # Original Pattern (torch.rms_norm)
                def forward(self, arg0_1: "bf16[1, 4096, 24, 128]", arg1_1: "bf16[128]"):
                    # File: /usr/local/python3.11.13/lib/python3.11/site-packages/torch/nn/functional.py:2925
                    # in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
                    _npu_dtype_cast: "f32[1, 4096, 24, 128]" = \
                        torch.ops.npu._npu_dtype_cast.default(arg0_1, torch.float32);  arg0_1 = None
                    pow_1: "f32[1, 4096, 24, 128]" = \
                        torch.ops.aten.pow.Tensor_Scalar(_npu_dtype_cast, 2)
                    mean: "f32[1, 4096, 24, 1]" = \
                        torch.ops.aten.mean.dim(pow_1, [3], True);  pow_1 = None
                    add: "f32[1, 4096, 24, 1]" = \
                        torch.ops.aten.add.Scalar(mean, 9.999999974752427e-07);  mean = None
                    rsqrt: "f32[1, 4096, 24, 1]" = \
                        torch.ops.aten.rsqrt.default(add);  add = None
                    mul: "f32[1, 4096, 24, 128]" = \
                        torch.ops.aten.mul.Tensor(_npu_dtype_cast, rsqrt); _npu_dtype_cast = rsqrt = None
                    mul_1: "f32[1, 4096, 24, 128]" = \
                        torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
                    _to_copy: "bf16[1, 4096, 24, 128]" = \
                        torch.ops.aten._to_copy.default(mul_1,
                            dtype = torch.bfloat16, 
                            layout = torch.strided, 
                            device = device(type='npu', index=0));  mul_1 = None
                    return (_to_copy,)
                '''
                input_dtype = hidden_states.dtype
                last_dim = hidden_states.dim() - 1
                
                hidden_states_fp32 = _dtype_cast_func(hidden_states, torch.float32)
                variance = hidden_states_fp32.pow(2).mean(last_dim, keepdim=True)

                if "2.6.0" in torch.__version__:
                    variance_eps = torch.ops.aten.add.Scalar(variance, _eps_in_bf16)
                    hidden_states_mul = hidden_states_fp32 * torch.rsqrt(variance_eps)
                    hidden_states_mul_cast = torch.ops.aten._to_copy.default(
                        hidden_states_mul, 
                        dtype=input_dtype, 
                        layout=torch.strided, 
                        device=hidden_states.device
                    )
                        
                    result = hidden_states_mul_cast * weight
                else:
                    variance_eps = torch.ops.aten.add.Scalar(variance, _eps_in_fp32)
                    hidden_states_mul = hidden_states_fp32 * torch.rsqrt(variance_eps)
                    hidden_states_mul_weight = hidden_states_mul * weight

                    result = torch.ops.aten._to_copy.default(
                        hidden_states_mul_weight, 
                        dtype=input_dtype, 
                        layout=torch.strided, 
                        device=hidden_states.device
                    )

                return result
                
            return func(hidden_states, weight)

        @staticmethod
        def replacement(hidden_states, weight):
            def func(hidden_states, weight):
                return torch_npu.npu_rms_norm(hidden_states, weight, epsilon=epsilon)[0]
            return func(hidden_states, weight)

    return RMSNormPattern

RMSNormPatternGroup = [create(dtype=torch.bfloat16, epsilon=1e-6)]
