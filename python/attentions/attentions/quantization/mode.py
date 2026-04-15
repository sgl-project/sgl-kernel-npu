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

from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Optional
import sys
from ..utils import ParametersInvalid
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum


class QuantAlgorithm(StrEnum):
    W8A8 = "W8A8"
    W8A8_TIMESTEP = "W8A8_TIMESTEP"
    W8A8_DYNAMIC = "W8A8_DYNAMIC"
    W4A4_DYNAMIC = "W4A4_DYNAMIC"
    W8A8_MXFP8 = "W8A8_MXFP8"
    W8A16 = "W8A16"
    W4A16 = "W4A16"
    W4A16_AWQ = "W4A16_AWQ"
    W4A8_AWQ = "W4A8_AWQ"
    W8A16_GPTQ = "W8A16_GPTQ"
    W4A16_GPTQ = "W4A16_GPTQ"
    W8A8_PER_CHANNEL = "W8A8_PER_CHANNEL"
    W8A8_PER_TENSOR = "W8A8_PER_TENSOR"
    W8A8_PER_CHANNEL_PER_TOKEN = "W8A8_PER_CHANNEL_PER_TOKEN"
    W8A8_PER_CHANNEL_PER_TENSOR = "W8A8_PER_CHANNEL_PER_TENSOR"
    W8A8_PER_TENSOR_PER_TOKEN = "W8A8_PER_TENSOR_PER_TOKEN"
    INT8 = "INT8"
    MIXED_PERCISION = "MIXED_PERCISION"
    FP8_DYNAMIC = "FP8_DYNAMIC"
    NO_QUANT = "NO_QUANT"
    W4A4_MXFP4_SVD = "W4A4_MXFP4_SVD"
    W4A4_MXFP4_DUALSCALE = "W4A4_MXFP4_DUALSCALE"


W8A8_LIST = [
    QuantAlgorithm.W8A8,
    QuantAlgorithm.W8A8_TIMESTEP,
    QuantAlgorithm.W8A8_DYNAMIC,
    QuantAlgorithm.W8A8_PER_CHANNEL,
    QuantAlgorithm.W8A8_PER_TENSOR,
    QuantAlgorithm.W8A8_PER_CHANNEL_PER_TOKEN,
    QuantAlgorithm.W8A8_PER_CHANNEL_PER_TENSOR,
    QuantAlgorithm.W8A8_PER_TENSOR_PER_TOKEN,
    QuantAlgorithm.W8A8_MXFP8,
]

W4A4_LIST = [
    QuantAlgorithm.W4A4_MXFP4_SVD,
    QuantAlgorithm.W4A4_MXFP4_DUALSCALE,
    QuantAlgorithm.W4A4_DYNAMIC,
]


@dataclass
class QuantModeDescriptor:
    quantize_weights: bool = field(default=False)
    quantize_activations: bool = field(default=False)
    per_token: bool = field(default=False)
    per_channel: bool = field(default=False)
    per_group: bool = field(default=False)
    use_int4_weights: bool = field(default=False)
    use_fa_quant: bool = field(default=False)


class QuantFlag(IntFlag):
    FA_QUANT = auto()

    INT4_WEIGHTS = auto()

    INT8_WEIGHTS = auto()

    ACTIVATION = auto()

    PER_CHANNEL = auto()

    PER_TENSOR = auto()

    PER_TOKEN = auto()

    PER_GROUP = auto()

    # 注意：这里是last auto， 后面不要添加auto
    COUNT = auto()

    WEIGHTS_AND_ACTIVATION = INT4_WEIGHTS | INT8_WEIGHTS | ACTIVATION

    # mask作用
    VALID_FLAG = COUNT - 1


class QuantMode():
    def __init__(self, flag: QuantFlag = 0):
        self.flag = flag

    def __deepcopy__(self, memodict=None):
        return self

    @staticmethod
    def from_descriptor(desc: QuantModeDescriptor):

        def raise_error(info: str):
            raise ParametersInvalid(f"Invalid quantization mode descriptor {desc}, err info:{info}")

        if desc.quantize_activations and not desc.quantize_weights:
            raise_error("To quantize activations, the weights must be quantized.")

        if (desc.per_token or desc.per_channel) and not (desc.quantize_weights and desc.quantize_activations):
            raise_error("To set per_token or per_channel, the activations and weights must be quantified.")

        mode = QuantMode()

        if desc.quantize_weights and desc.use_int4_weights:
            mode.flag |= QuantFlag.INT4_WEIGHTS
        elif desc.quantize_weights:
            mode.flag |= QuantFlag.INT8_WEIGHTS

        if desc.quantize_activations:
            mode.flag |= QuantFlag.ACTIVATION

        if desc.per_channel:
            mode.flag |= QuantFlag.PER_CHANNEL
        if desc.per_token:
            mode.flag |= QuantFlag.PER_TOKEN
        if desc.per_group:
            mode.flag |= QuantFlag.PER_GROUP
        if desc.use_fa_quant:
            mode.flag |= QuantFlag.FA_QUANT
        return mode

    @staticmethod
    def use_smooth_quant(per_token=False, per_channel=False):
        desc = QuantModeDescriptor()
        desc.per_token = per_token
        desc.per_channel = per_channel
        desc.quantize_weights = True
        desc.quantize_activations = True
        return QuantMode.from_descriptor(desc)

    @staticmethod
    def use_weight_only(use_int4_weights=False, per_group=False):
        desc = QuantModeDescriptor()
        desc.use_int4_weights = use_int4_weights
        desc.per_group = per_group
        desc.quantize_weights = True
        desc.quantize_activations = False
        desc.per_token = False
        desc.per_channel = False
        return QuantMode.from_descriptor(desc)

    @staticmethod
    def from_quant_algo(quant_algo: Optional[QuantAlgorithm] = None):
        quant_mode_map = {
            QuantAlgorithm.W8A16: QuantMode.use_weight_only(use_int4_weights=False),
            QuantAlgorithm.W4A16: QuantMode.use_weight_only(use_int4_weights=True),
            QuantAlgorithm.W4A16_AWQ: QuantMode.use_weight_only(use_int4_weights=True, per_group=True),
            QuantAlgorithm.W4A8_AWQ: QuantMode.use_weight_only(use_int4_weights=True, per_group=True),
            QuantAlgorithm.W4A16_GPTQ: QuantMode.use_weight_only(use_int4_weights=True, per_group=True),
            QuantAlgorithm.W8A16_GPTQ: QuantMode.use_weight_only(use_int4_weights=False, per_group=True),
            QuantAlgorithm.W8A8_PER_CHANNEL: QuantMode.use_smooth_quant(per_token=False, per_channel=True),
            QuantAlgorithm.W8A8_PER_TENSOR: QuantMode.use_smooth_quant(per_token=False, per_channel=False),
            QuantAlgorithm.W8A8_PER_CHANNEL_PER_TENSOR: QuantMode.use_smooth_quant(per_token=False,
                                                                                      per_channel=True),
            QuantAlgorithm.W8A8: QuantMode.use_smooth_quant(per_token=False, per_channel=False),
            QuantAlgorithm.W8A8_TIMESTEP: QuantMode.use_smooth_quant(per_token=False, per_channel=False),
            QuantAlgorithm.W8A8_DYNAMIC: QuantMode.use_smooth_quant(per_token=False, per_channel=False),
            QuantAlgorithm.W4A4_DYNAMIC: QuantMode.use_smooth_quant(per_token=True, per_channel=True),
            QuantAlgorithm.W8A8_PER_CHANNEL_PER_TOKEN: QuantMode.use_smooth_quant(per_token=True, per_channel=True),
            QuantAlgorithm.W8A8_PER_TENSOR_PER_TOKEN: QuantMode.use_smooth_quant(per_token=True, per_channel=False),
            QuantAlgorithm.FP8_DYNAMIC: QuantMode.from_descriptor(QuantModeDescriptor(use_fa_quant=True)),
            QuantAlgorithm.W8A8_MXFP8: QuantMode.use_smooth_quant(per_token=False, per_channel=False),
            QuantAlgorithm.W4A4_MXFP4_SVD: QuantMode.use_smooth_quant(per_token=False, per_channel=False),
            QuantAlgorithm.W4A4_MXFP4_DUALSCALE: QuantMode.use_smooth_quant(per_token=False, per_channel=False),
        }
        return quant_mode_map.get(quant_algo, QuantMode(0))

    def check_weight_int8_only(self):
        return self._all(QuantFlag.INT8_WEIGHTS, QuantFlag.WEIGHTS_AND_ACTIVATION)
        
    def contains_fa_quantization(self):
        return self._any(QuantFlag.FA_QUANT)
        
    def contains_per_group_scale(self):
        return self._any(QuantFlag.PER_GROUP)
        
    def contains_weight_quantization(self):
        return self._any(QuantFlag.INT4_WEIGHTS | QuantFlag.INT8_WEIGHTS)
        
    def check_weight_int4_only(self):
        return self._all(QuantFlag.INT4_WEIGHTS, QuantFlag.WEIGHTS_AND_ACTIVATION)
        
    def check_weight_only_mode(self):
        return self.check_weight_int8_only() or self.check_weight_int4_only()
        
    def contains_activation_or_weight_quant(self):
        return self._any(QuantFlag.INT4_WEIGHTS | QuantFlag.INT8_WEIGHTS | QuantFlag.ACTIVATION)
        
    def check_weight_int8_only_with_group(self):
        return self.check_weight_int8_only() and self._any(QuantFlag.PER_GROUP)
        
    def contains_per_channel_scale(self):
        return self._any(QuantFlag.PER_CHANNEL)
        
    def contains_activation_and_weight_quant(self):
        return self._all(QuantFlag.INT8_WEIGHTS | QuantFlag.ACTIVATION, QuantFlag.WEIGHTS_AND_ACTIVATION)
        
    def check_weight_int4_only_with_group(self):
        return self.check_weight_int4_only() and self._any(QuantFlag.PER_GROUP)


    def to_dict(self):
        return {
            "use_smooth_quant": self.contains_activation_and_weight_quant(),
            "use_weight_only": self.check_weight_only_mode(),
            "weight_only_precision": 'int8' if self.check_weight_int8_only() else 'int4',
        }

    def _all(self, bits, mask=QuantFlag.VALID_FLAG):
        return (self.flag & mask) == bits

    def _any(self, bits):
        return (self.flag & bits) != 0
