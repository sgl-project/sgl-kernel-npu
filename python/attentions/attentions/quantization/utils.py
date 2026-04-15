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
from typing import Optional
import contextvars
from inspect import signature, Parameter
from torch import distributed as dist
from ..utils import ParametersInvalid, ConfigError
from ..utils.logs.logging import logger

MAX_WEIGHT_SIZE = 100 * 1024 * 1024 * 1024 # 工具对SD量化还不能分片保存


def extract_constructor_args(instance, base_class=None):
    cls = instance.__class__
    init_params = signature(cls.__init__).parameters
    if not init_params:
        raise ParametersInvalid(f"init_params is none!")
    param_names = [k for k, v in init_params.items() if v.kind == Parameter.POSITIONAL_OR_KEYWORD and k != 'self']

    if base_class:
        base_params = signature(base_class.__init__).parameters
        base_param_names = {k for k in base_params if k != 'self'}
        param_names = [n for n in param_names if n in base_param_names]

    return {name: getattr(instance, name) for name in param_names
            if hasattr(instance, name)}


def replace_rank_suffix(file_path):
    # 分离目录路径和文件名（处理多级目录）
    dir_path, filename = os.path.split(file_path)
    # 分离主文件名和扩展名（如 .json）
    basename, ext = os.path.splitext(filename)

    # 按最后一个下划线分割文件名
    parts = basename.rsplit('_', 1)

    rank = -1

    # 检查后缀是否为数字
    if len(parts) > 1 and parts[-1].isdigit():
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            raise ConfigError(f"must init distributed env if use distributed config {filename}")
        new_basename = f"{parts[0]}_{rank}"
    else:
        new_basename = basename  # 不修改原文件名

    # 重组路径
    new_filename = f"{new_basename}{ext}"
    new_path = os.path.join(dir_path, new_filename)
    return new_path, new_filename, rank


def get_quant_weight(weights, key):
    """安全获取量化偏置张量并转换NPU格式
    Args:
        weights (dict): 参数字典
        key (str): 参数前缀标识符

    Returns:
        torch.Tensor: 转换后的张量

    Raises:
        KeyError: 当指定键不存在时抛出
    """
    if key in weights.keys():
        tensor = weights.get_tensor(key)
    else:
        raise ParametersInvalid(f"Critical parameter missing: {key}.")
    return tensor


class TimestepManager:
    """Manages timestep indices for multi-modal quantization processes."""

    _timestep_var = contextvars.ContextVar("timestep_idx", default=None)
    _timestep_var_max = contextvars.ContextVar("timestep_idx_max", default=None)

    @classmethod
    def set_timestep_idx(cls, cur_timestep: int) -> None:
        r"""
        The method is used to set the current timestep.

        Args:
            cur_timestep: Current iteration timestep.
        """
        if cur_timestep is not None and not isinstance(cur_timestep, int):
            raise ParametersInvalid(f"cur_timestep must be the type of int, but currently got {type(cur_timestep)}.")
        current = cls._timestep_var.get()
        max_step = cls._timestep_var_max.get()
        if current is not None and current == cur_timestep:
            logger.debug("Warning: Setting same timestep value consecutively: %r", cur_timestep)
        if max_step is not None and cur_timestep > max_step:
            raise ParametersInvalid(f"max timestep set in quant weight: {max_step}.")
        cls._timestep_var.set(cur_timestep)
        logger.debug("Timestep index set to: %r", cur_timestep)

    @classmethod
    def get_timestep_idx(cls) -> Optional[int]:
        r"""
        Get the current timestep index.
        Returns:
            The current timestep index.
        """
        t_idx = cls._timestep_var.get()
        if t_idx is None:
            logger.debug("Warning: Timestep index not set. Call set_timestep_idx() before each timestep.")
        return t_idx
    
    @classmethod
    def set_timestep_idx_max(cls, t_idx: int) -> None:
        r"""
        Set the max timestep index.
        Args:
            t_idx: The max current timestep index.
        """
        if t_idx is not None and not isinstance(t_idx, int):
            raise ParametersInvalid(f"t_idx must be the type of int, but currently got {type(t_idx)}.")
        current = cls._timestep_var_max.get()
        if current is not None and current == t_idx:
            logger.debug("Warning: Setting same Max timestep value consecutively: %r", t_idx)
        cls._timestep_var_max.set(t_idx)
        logger.debug("Max Timestep index set to: %r", t_idx)

    @classmethod
    def get_timestep_idx_max(cls) -> Optional[int]:
        r"""
        Get the max timestep index.
        Returns:
            The max current timestep index.
        """
        t_idx = cls._timestep_var_max.get()
        if t_idx is None:
            logger.debug("Warning: Max Timestep index not set. "
                "Call set_timestep_idx_max() before get_timestep_idx_max.")
        return t_idx
