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

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch_npu

from .config import TimestepPolicyConfig
from .utils import get_quant_weight, TimestepManager


class WeightQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super(WeightQuantLinear, self).__init__()
        # 根据入参作为可选属性
        self.prefix = prefix
        self.dtype = kwargs.get('dtype', torch.bfloat16)
        self.input_feature = in_features
        self.output_feature = out_features
    
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale').T.to(self.dtype)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        
        weight = get_quant_weight(weights, f'{prefix}.weight')
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight, 29).T
            if kwargs.get('is_w4', False):
                weight = torch_npu.npu_convert_weight_to_int4pack(weight.npu().to(torch.int32))
        else:
            weight = weight.T
            if kwargs.get('is_w4', False):
                weight = torch_npu.npu_convert_weight_to_int4pack(weight.npu().to(torch.int32))
        self.register_buffer("weight", weight, persistent=False)
        
        if bias:
            bias = get_quant_weight(weights, f'{prefix}.bias')
            if self.dtype == torch.bfloat16:
                bias = bias.to(torch.float32)
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        output = torch_npu.npu_weight_quant_batchmatmul(x, self.weight, self.weight_scale,
            bias=self.bias)
        return output

    def forward(self, x):
        # dynamic场景算子虽然也支持3维，但性能会劣化，这里展平做运算
        if x.ndim >= 3:
            return self._flatten_linear(x)
        output = self.quant_matmul(x)
        return output

    def _flatten_linear(self, x):
        x_reshpe = x.reshape(x.shape[:-1].numel(), -1)
        output = self.quant_matmul(x_reshpe)
        new_size = list(x.shape)[:-1]
        new_size.append(output.shape[1])
        return output.view(*new_size)


class W8A8QuantBaseLinear(ABC, nn.Module):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__()
        # 根据入参作为可选属性
        self.dtype = kwargs.get('dtype', torch.bfloat16)
        self.input_feature = in_features
        self.output_feature = out_features

        if bias:
            bias = get_quant_weight(weights, f'{prefix}.bias')
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None
        mul_scale = kwargs.get('mul_scale', None)
        if mul_scale is not None:
            mul_scale = mul_scale.to(self.dtype)
            self.register_buffer("mul_scale", mul_scale, persistent=False)
        else:
            self.mul_scale = None
    
    def pack_weight(self, weight, **kwargs):
        return weight

    @abstractmethod
    def quant_matmul(self, x):
        pass

    def forward(self, x):
        # dynamic场景算子虽然也支持3维，但性能会劣化，这里展平做运算
        if x.ndim >= 3 or (x.ndim == 3 and self.is_dynamic):
            return self._flatten_linear(x)
        output = self.quant_matmul(x)
        return output

    def _flatten_linear(self, x):
        x_reshpe = x.reshape(x.shape[:-1].numel(), -1)
        output = self.quant_matmul(x_reshpe)
        new_size = list(x.shape)[:-1]
        new_size.append(output.shape[1])
        return output.view(*new_size)

    def _init_static_quant_param(self, prefix=None, weights=None, **kwargs):
        quant_bias = get_quant_weight(weights, f'{prefix}.quant_bias')
        self.register_buffer("quant_bias", quant_bias, persistent=False)
        deq_scale = get_quant_weight(weights, f'{prefix}.deq_scale')
        self.register_buffer("deq_scale", deq_scale, persistent=False)
        weight = get_quant_weight(weights, f'{prefix}.weight')
        self.register_buffer("weight", weight, persistent=False)
        if self.dtype == torch.float16:
            input_scale = get_quant_weight(weights, f'{prefix}.input_scale').to(torch.float32)
        else:
            input_scale = get_quant_weight(weights, f'{prefix}.input_scale').to(self.dtype)
        if input_scale.dim() == 1:
            input_scale = input_scale.repeat(weight.data.shape[1])
        else:
            input_scale = input_scale.repeat(1, weight.data.shape[1])

        self.register_buffer("input_scale", input_scale, persistent=False)
        
        input_offset = get_quant_weight(weights, f'{prefix}.input_offset').to(torch.int8)
        if input_offset.dim() == 1:
            input_offset = input_offset.repeat(weight.data.shape[1])
        else:
            input_offset = input_offset.repeat(1, weight.data.shape[1])

        self.register_buffer("input_offset", input_offset, persistent=False)
        
    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale').squeeze().to(self.dtype)
        if self.dtype == torch.float16:
            weight_scale = weight_scale.to(torch.float32)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        if self.bias is not None:
            self.bias = self.bias.to(self.dtype)
        weight = get_quant_weight(weights, f'{prefix}.weight')
        weight = self.pack_weight(weight, **kwargs)
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight.npu(), 29).T
        else:
            weight = weight.T
        self.register_buffer("weight", weight, persistent=False)


class W8A8QuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.is_dynamic = kwargs.get('is_dynamic', False)
        
        if not self.is_dynamic:
            self._init_static_quant_param(prefix, weights, **kwargs)
        else:
            self._init_dynamic_quant_param(prefix, weights, **kwargs)


    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if not self.is_dynamic:
            if self.mul_scale is not None:
                x_scaled = x * self.mul_scale
                x_int8 = torch_npu.npu_quantize(x_scaled, scales=self.input_scale,
                    zero_points=self.input_offset, dtype=torch.qint8, axis=-1)
            else:
                x_int8 = torch_npu.npu_quantize(x, scales=self.input_scale,
                    zero_points=self.input_offset, dtype=torch.qint8, axis=-1)

            output = torch_npu.npu_quant_matmul(x_int8, self.weight.T, self.deq_scale, 
                                                bias=self.quant_bias, 
                                                output_dtype=self.dtype)
        else:
            if self.mul_scale is not None:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x * self.mul_scale)
            else:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x)

            output = torch_npu.npu_quant_matmul(x_int8, self.weight, self.weight_scale,
                                                pertoken_scale=input_scale, output_dtype=self.dtype,
                                                bias=self.bias)
        return output


class W4A4QuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)
        self.is_dynamic = True
        self._init_dynamic_quant_param(prefix, weights, **kwargs)
    
    def pack_weight(self, weight, **kwargs):
        weight.data = torch_npu.npu_convert_weight_to_int4pack(weight.data.to(torch.int32).npu())
        return weight

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
        pertoken_scale = pertoken_scale.reshape(-1, 1)
        pertoken_scale = pertoken_scale.squeeze(-1)
        output = torch_npu.npu_quant_matmul(
            x, self.weight, self.weight_scale.data.view(-1), 
            pertoken_scale=pertoken_scale, bias=None, output_dtype=self.dtype
            )
        return output


class W8A8TimeStepQuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.timestep_config = kwargs.get('timestep_config', TimestepPolicyConfig())

        self.is_dynamic = True

        self._init_dynamic_quant_param(prefix, weights, **kwargs)
        # 最后使用的是n k的权重
        self._init_static_quant_param(prefix, weights, **kwargs)
        
        TimestepManager.set_timestep_idx_max(self.input_scale.shape[0])

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        # 判断时间步状态
        t_idx = TimestepManager.get_timestep_idx()
        strategy = self.timestep_config.get_strategy(t_idx)

        if strategy == "static":
            self.is_dynamic = False
        else:
            self.is_dynamic = True

        if not self.is_dynamic:
            if self.mul_scale is not None:
                x_scaled = x * self.mul_scale
                x_int8 = torch_npu.npu_quantize(x_scaled, scales=self.input_scale[t_idx],
                    zero_points=self.input_offset[t_idx], dtype=torch.qint8, axis=-1)
            else:
                x_int8 = torch_npu.npu_quantize(x, scales=self.input_scale[t_idx],
                    zero_points=self.input_offset[t_idx], dtype=torch.qint8, axis=-1)

            output = torch_npu.npu_quant_matmul(x_int8, self.weight.T, self.deq_scale[t_idx],
                                                bias=self.quant_bias[t_idx],
                                                output_dtype=self.dtype)
        else:
            if self.mul_scale is not None:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x * self.mul_scale)
            else:
                x_int8, input_scale = torch_npu.npu_dynamic_quant(x)

            output = torch_npu.npu_quant_matmul(x_int8, self.weight.T, self.weight_scale,
                                                pertoken_scale=input_scale, output_dtype=self.dtype,
                                                bias=self.bias)
        return output


class FP8RotateQuantFA(nn.Module):
    def __init__(self, prefix=None, weights=None):
        super().__init__()

        q_rot = get_quant_weight(weights, f'{prefix}.q_rot')
        self.register_buffer("q_rot", q_rot, persistent=False)
        k_rot = get_quant_weight(weights, f'{prefix}.k_rot')
        self.register_buffer("k_rot", k_rot, persistent=False)

    def forward(self, query, key, value, **kwargs):
        query = torch.matmul(query, self.q_rot)
        key = torch.matmul(key, self.k_rot)

        layout = kwargs.get("layout", "BNSD")

        from ..layers.quant.block_quant import fa_block_quant_preprocess

        q, q_scale = fa_block_quant_preprocess(query, block_size=128,
                                               dst_type=torch_npu.float8_e4m3fn, layout=layout)
        k, k_scale = fa_block_quant_preprocess(key, block_size=256,
                                               dst_type=torch_npu.float8_e4m3fn, layout=layout)
        v, v_scale = fa_block_quant_preprocess(value, block_size=256,
                                               dst_type=torch_npu.float8_e4m3fn, layout=layout)

        if layout == "BNSD":
            _, n, s, d = query.shape
        elif layout == "BSND":
            _, s, n, d = query.shape
    
        x = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, input_layout=layout,
                                                            num_query_heads=n,
                                                            softmax_scale=1.0 / math.sqrt(d),
                                                            pre_tokens=2147483647,
                                                            next_tokens=2147483647, 
                                                            query_quant_mode=7,
                                                            key_quant_mode=7,
                                                            value_quant_mode=7,
                                                            dequant_scale_query=q_scale,
                                                            dequant_scale_key=k_scale,
                                                            dequant_scale_value=v_scale,
                                                            out_dtype=query.dtype
                                                            )[0]
        
        if x.shape[2] != s:
            if layout == "BNSD":
                x = x[:, :, :s, :]
            elif layout == "BSND":
                x = x[:, :s, :, :]

        return x


class W8A8MXFP8QuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.is_dynamic = True
        self._init_dynamic_quant_param(prefix, weights, **kwargs)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if self.mul_scale is not None:
            x1, input_scale = torch_npu.npu_dynamic_mx_quant(x * self.mul_scale, dst_type=torch_npu.float8_e4m3fn)
        else:
            x1, input_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)

        if self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)

        x2 = self.weight
        if x2.dtype != torch.float8_e4m3fn:
            x2 = torch_npu.npu_dtype_cast(x2, torch_npu.float8_e4m3fn)
        x2 = x2.transpose(0, 1)

        output = torch_npu.npu_quant_matmul(
                            x1,
                            x2,
                            self.weight_scale.transpose(0, 1),
                            scale_dtype=torch_npu.float8_e8m0fnu,
                            pertoken_scale=input_scale,
                            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                            bias=self.bias,
                            output_dtype=self.dtype,
                            group_sizes=[1, 1, 32],
                            )
        return output

    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale')
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        weight = get_quant_weight(weights, f'{prefix}.weight')
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight.npu(), 29)
        self.register_buffer("weight", weight, persistent=False)


class W4A4MXFP4DualQuantLinear(W8A8QuantBaseLinear):
    def __init__(self, in_features, out_features, bias=True, weights=None, prefix=None, **kwargs):
        super().__init__(in_features, out_features, bias, weights, prefix, **kwargs)

        self.is_dynamic = kwargs.get('is_dynamic', True)
        self._init_dynamic_quant_param(prefix, weights, **kwargs)

    def quant_matmul(self, x):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if self.mul_scale is not None:
            x1, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(x * self.mul_scale, smooth_scale=None)
        else:
            x1, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=None)

        if self.bias.dtype != torch.float32:
            self.bias = self.bias.to(torch.float32)

        output = torch_npu.npu_dual_level_quant_matmul(x1,
                                                    self.weight,
                                                    l0_scale,
                                                    self.weight_dual_scale,
                                                    l1_scale,
                                                    self.weight_scale,
                                                    bias=self.bias,
                                                    output_dtype=self.dtype)
        return output

    def _init_dynamic_quant_param(self, prefix=None, weights=None, **kwargs):
        weight_scale = get_quant_weight(weights, f'{prefix}.weight_scale')
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        weight_dual_scale = get_quant_weight(weights, f'{prefix}.weight_dual_scale')
        weight_dual_scale = weight_dual_scale.squeeze(-1).transpose(0, 1).contiguous()
        self.register_buffer("weight_dual_scale", weight_dual_scale, persistent=False)

        weight = get_quant_weight(weights, f'{prefix}.weight')
        weight = torch_npu.npu_dtype_cast(weight.npu(), torch_npu.float4_e2m1fn_x2)
        if kwargs.get('use_nz', False):
            weight = torch_npu.npu_format_cast(weight.view(torch.int8), 29, torch.int8)
        self.register_buffer("weight", weight, persistent=False)