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
import dataclasses
from functools import cached_property, wraps
from typing import Dict, Optional, Tuple

from .mode import QuantAlgorithm, QuantMode
from ..utils import ModelInitError, ParametersInvalid

VALID_STRATEGIES = ["dynamic", "static"]


def validate_quant_config_init_params(func):
    @wraps(func)
    def wrapper(self):
        # 验证量化算法
        if self.quant_algo is not None and not isinstance(self.quant_algo, QuantAlgorithm):
            raise ModelInitError(
                f'self.quant_algo must be an instance of QuantAlgorithm, but actually got {type(self.quant_algo)}.')

        # 验证排除层
        if self.exclude_layers is not None:
            if not isinstance(self.exclude_layers, tuple):
                raise ModelInitError("self.exclude_layers must be a tuple")
            for layer in self.exclude_layers:
                if not isinstance(layer, str):
                    raise ModelInitError("Items in exclude_layers must be strings")

        # 调用原始函数
        result = func(self)

        return result

    return wrapper


@dataclasses.dataclass
class QuantConfig:
    quant_algo: Optional[QuantAlgorithm] = None
    exclude_layers: Optional[Tuple[str]] = None

    @validate_quant_config_init_params
    def __post_init__(self):
        pass

    @classmethod
    def parse_from_dict(cls, config: dict):
        obj = cls(**config)
        return obj

    @cached_property
    def layer_quantization_mode(self) -> QuantMode:
        return QuantMode.from_quant_algo(self.quant_algo)

    def serialize_to_dict(self):
        return dataclasses.asdict(self)


def validate_layer_quant_config_init_params(func):
    @wraps(func)
    def wrapper(self):
        # 验证量化算法
        if self.quant_algo is not None and not isinstance(self.quant_algo, QuantAlgorithm):
            raise ModelInitError(
                f'self.quant_algo must be an instance of QuantAlgorithm, but actually got {type(self.quant_algo)}.')

        # 验证量化层
        if self.quantized_layers is not None:
            if not isinstance(self.quantized_layers, dict):
                raise ModelInitError("self.quantized_layers must be a dictionary")
            for name, layer_config in self.quantized_layers.items():
                if not isinstance(name, str):
                    raise ModelInitError("Keys in self.quantized_layers must be strings")
                if not isinstance(layer_config, QuantConfig):
                    raise ModelInitError("Values in self.quantized_layers must be instances of QuantConfig")

        # 验证排除层
        if self.exclude_layers is not None:
            if not isinstance(self.exclude_layers, tuple):
                raise ModelInitError("self.exclude_layers must be a tuple")
            for layer in self.exclude_layers:
                if not isinstance(layer, str):
                    raise ModelInitError("Items in self.exclude_layers must be strings")

        # 调用原始函数
        result = func(self)

        return result

    return wrapper


@dataclasses.dataclass
class LayerQuantConfig(QuantConfig):
    quant_algo: Optional[QuantAlgorithm] = None
    quantized_layers: Optional[Dict[str, QuantConfig]] = None
    exclude_layers: Optional[Tuple[str]] = None

    @validate_layer_quant_config_init_params
    def __post_init__(self):
        self.auto_quant_mode = {}
        if self.quantized_layers:
            self.auto_quant_mode = {
                name: QuantMode.from_quant_algo(layer_config.quant_algo)
                for name, layer_config in self.quantized_layers.items()
            }

    @classmethod
    def parse_from_dict(cls, config: dict):
        # 提取并处理量化层配置
        quantized_layers = config.pop('quantized_layers', {})
        quantized_layers_dict = {
            name: QuantConfig.parse_from_dict(layer_config)
            for name, layer_config in quantized_layers.items()
        }
        
        return cls(quantized_layers=quantized_layers_dict, **config)

    @cached_property
    def layer_quantization_mode(self) -> Dict[str, QuantMode]:
        return self.auto_quant_mode

    @cached_property
    def quant_algorithms_list(self):
        if not self.quantized_layers:
            return []
        return list(set(layer_config.quant_algo for _, layer_config in self.quantized_layers.items()))


    def serialize_to_dict(self):
        # 创建输出字典，只包含需要序列化的字段
        quant_layer = "quantized_layers"
        output = {
            'quant_algo': self.quant_algo,
            quant_layer: {}
        }
        
        # 处理量化层配置
        if self.quantized_layers:
            output[quant_layer] = {
                name: layer_config.serialize_to_dict()
                for name, layer_config in self.quantized_layers.items()
            }
            
            # 移除每个层配置中的exclude_layers字段
            for layer_dict in output[quant_layer].values():
                layer_dict.pop('exclude_layers', None)
                
        return output


class TimestepPolicyConfig:
    def __init__(self):
        r"""
        The method is used to init TimestepPolicyConfig.
        """
        self._strategies = {}  # 策略注册表
        self._default_strategy = "dynamic"

    def register(self, step_range, strategy):
        r"""
        The method is used to register strategy.

        Args:
            step_range: Timestep range, the type can be int, range, or list.
            strategy: Supports two types of strings, static and dynamic, to represent two strategy.
        """
        if not isinstance(strategy, str):
            raise TypeError(f"strategy_cls必须是字符串类型，实际类型：{type(strategy)}")
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"无效的策略类型：{type(strategy)}，允许值：{VALID_STRATEGIES}")
        if isinstance(step_range, int):
            step_range = [step_range]
        elif isinstance(step_range, (list, range)):
            if not all(isinstance(s, int) for s in step_range):
                raise TypeError("step_range列表必须包含整数元素")
        else:
            raise TypeError(f"step_range必须是int, list, range类型，实际类型：{type(step_range)}")
        for step in step_range:
            self._strategies[step] = strategy

    def get_strategy(self, step):
        r"""
        The method is used to get strategy.

        Args:
            step: Specifild timestep.
        Returns:
            The strategy corresponding to the specifiled timestep.
        """
        if step is not None and not isinstance(step, int):
            raise ParametersInvalid(f"step must be the type of int, but currently got {type(step)}.")
        return self._strategies.get(step, self._default_strategy)
