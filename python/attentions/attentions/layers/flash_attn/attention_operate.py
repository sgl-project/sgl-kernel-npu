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

from abc import ABC, abstractmethod
import torch

from .common import AttentionParam
from ...utils.exception import ParametersInvalid


class AttentionOperateBase(ABC):
    layout_to_func = {
        "BNSD": "forward_attn_bnsd",
        "BSND": "forward_attn_bsnd",
        "BSH": "forward_attn_bsh"}
    supported_layout = None
    supported_dtype = None
    
    @classmethod
    @abstractmethod
    def is_supported_shape(cls, attn_param: AttentionParam) -> bool:
        pass
    
    @classmethod
    def is_supported_layout(cls, layout: str) -> bool:
        if cls.supported_layout is None:
            raise ParametersInvalid("The supported_layout is not initialized in subclasses.")
        return layout in cls.supported_layout

    @classmethod
    def is_supported_dtype(cls, dtype: torch.dtype) -> bool:
        if cls.supported_dtype is None:
            raise ParametersInvalid("The supported_dtype is not initialized in subclasses.")
        return dtype in cls.supported_dtype
    
    @classmethod
    @abstractmethod
    def forward_attn_bnsd(cls, attn_param, query, key, value, mask=None, scale=None) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    @abstractmethod
    def forward_attn_bsnd(cls, attn_param, query, key, value, mask=None, scale=None) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    @abstractmethod
    def forward_attn_bsh(cls, attn_param, query, key, value, mask=None, scale=None) -> None:
        raise NotImplementedError("Subclasses must implement this method")


class AttnOpRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str, obj):
        if not isinstance(name, str):
            raise ParametersInvalid("Name must be a string")
        if not callable(obj):
            raise ParametersInvalid("Object must be callable")
        self._registry[name] = obj

    def get_all(self):
        return self._registry

    def get(self, name: str):
        try:
            return self._registry[name]
        except Exception as e:
            raise ParametersInvalid(f"Cannot find op: {name} in op registry.") from e


device_duo_op = AttnOpRegistry()
device_800_op = AttnOpRegistry()
device_a5_op = AttnOpRegistry()


def register_op_duo(name: str):
    def decorator(obj):
        device_duo_op.register(name, obj)
        return obj

    return decorator


def register_op_800(name: str):
    def decorator(obj):
        device_800_op.register(name, obj)
        return obj

    return decorator


def register_op_a5(name: str):
    def decorator(obj):
        device_a5_op.register(name, obj)
        return obj

    return decorator