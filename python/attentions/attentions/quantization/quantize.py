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

import json
import os
from typing import Dict
from collections import OrderedDict
from functools import wraps
import torch
import torch.nn as nn
import safetensors
from .mode import QuantAlgorithm
from .config import QuantConfig, LayerQuantConfig, TimestepPolicyConfig
from .mode import W4A4_LIST,W8A8_LIST
from .utils import replace_rank_suffix, get_quant_weight, extract_constructor_args, MAX_WEIGHT_SIZE
from .layer import (W4A4QuantLinear, W4A4MXFP4DualQuantLinear, W8A8QuantLinear, W8A8TimeStepQuantLinear,
                    WeightQuantLinear, FP8RotateQuantFA, W8A8MXFP8QuantLinear)
from ..utils import ParametersInvalid, ConfigError
from ..utils import file_utils
from ..utils.logs.logging import logger


def get_key_patterns(layer_name):
    key_patterns = [
        f'{layer_name}.linear.weight', 
        f'{layer_name}.weight', 
        f'{layer_name}', 
        f'{layer_name}.fa_q.scale',
        f'{layer_name}.quant_type'
    ]
    return key_patterns


def weight_quantize(name, layer, cfg, quant_weights, **kwargs):
    if cfg.quant_algo in [QuantAlgorithm.W8A16, QuantAlgorithm.W4A16]:
        return w8a16_quantize(name, layer, cfg, quant_weights, **kwargs)
    return layer, False


def w8a16_quantize(name, layer, cfg, quant_weights, **kwargs):
    quant_map = OrderedDict([
        (nn.Linear, WeightQuantLinear)
    ])

    # 如果模型指定了类的匹配规则，优先匹配模型指定的
    user_dict = kwargs.get('map', None)
    if user_dict:
        for key, value in user_dict.items():
            quant_map[key] = value
        for key in user_dict.keys():
            quant_map.move_to_end(key, last=False)

    # 寻找匹配的规则
    quant_cls = next((quant_map[cls] for cls in quant_map if isinstance(layer, cls)), None)
    
    if quant_cls is None:
        return layer, False

    # 获取浮点类的入参
    init_params = extract_constructor_args(layer, quant_cls)
    bias = 'bias'
    if bias in init_params and isinstance(init_params[bias], nn.Parameter):
        init_params[bias] = True
    else:
        init_params[bias] = False

    init_params['weights'] = quant_weights
    init_params['prefix'] = name
    if cfg.quant_algo == QuantAlgorithm.W4A16:
        init_params['is_w4'] = True

    # 抑制算法需要的属性    
    if f'{name}.div.mul_scale' in quant_weights.keys():
        init_params['mul_scale'] = get_quant_weight(quant_weights, f'{name}.div.mul_scale')
        init_params['prefix'] = f'{name}.linear'

    del layer.weight
    if hasattr(layer, 'bias'):
        del layer.bias
    quant_layer = quant_cls(**init_params, **kwargs)

    return quant_layer, True


def smooth_quantize_w8a8(name, layer, cfg, quant_weights, **kwargs):
    if cfg.quant_algo == QuantAlgorithm.W8A8_TIMESTEP:
        quant_map = OrderedDict([(nn.Linear, W8A8TimeStepQuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W8A8_MXFP8:
        quant_map = OrderedDict([(nn.Linear, W8A8MXFP8QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_DYNAMIC:
        quant_map = OrderedDict([(nn.Linear, W4A4QuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_DUALSCALE:
        quant_map = OrderedDict([(nn.Linear, W4A4MXFP4DualQuantLinear)])
    elif cfg.quant_algo == QuantAlgorithm.W4A4_MXFP4_SVD:
        raise ParametersInvalid("SVD Quant algorithm not supported!")
    else:
        quant_map = OrderedDict([(nn.Linear, W8A8QuantLinear)])

    # 如果模型指定了类的匹配规则，优先匹配模型指定的
    user_dict = kwargs.get('map', None)
    if user_dict:
        for key, value in user_dict.items():
            quant_map[key] = value
        for key in user_dict.keys():
            quant_map.move_to_end(key, last=False)

    # 寻找匹配的规则
    quant_cls = next((quant_map[cls] for cls in quant_map if isinstance(layer, cls)), None)

    if quant_cls is None:
        return layer, False

    # 获取浮点类的入参
    init_params = extract_constructor_args(layer, quant_cls)
    bias = 'bias'
    if bias in init_params and isinstance(init_params[bias], nn.Parameter):
        init_params[bias] = True
    else:
        init_params[bias] = False

    if cfg.quant_algo in [QuantAlgorithm.W8A8_DYNAMIC, QuantAlgorithm.W8A8_MXFP8, QuantAlgorithm.W4A4_DYNAMIC,
                          QuantAlgorithm.W4A4_MXFP4_DUALSCALE]:
        init_params['is_dynamic'] = True

    init_params['weights'] = quant_weights
    init_params['prefix'] = name
    # 抑制算法需要的属性    
    if f'{name}.div.mul_scale' in quant_weights.keys():
        init_params['mul_scale'] = get_quant_weight(quant_weights, f'{name}.div.mul_scale')
        init_params['prefix'] = f'{name}.linear'

    del layer.weight
    if hasattr(layer, 'bias'):
        del layer.bias

    quant_layer = quant_cls(**init_params, **kwargs)

    return quant_layer, True


def smooth_quantize(name, layer, cfg, quant_weights, **kwargs):
    if cfg.quant_algo in W8A8_LIST or cfg.quant_algo in W4A4_LIST:
        return smooth_quantize_w8a8(name, layer, cfg, quant_weights, **kwargs)
    return layer, False


def add_fa_quant(layer, cfg, prefix, quant_weights):
    if cfg.quant_algo in [QuantAlgorithm.FP8_DYNAMIC]:
        layer.fa_quant = FP8RotateQuantFA(prefix, quant_weights)
    return


def get_layer_quant_mode(name, layer, cfg):
    layer_quant_mode = None
    
    for pattern in get_key_patterns(name):
        if pattern in cfg.layer_quantization_mode:
            return cfg.layer_quantization_mode[pattern]
    return layer_quant_mode


def get_layer_quant_cfg(cfg, name, layer):
    layer_quant_cfg = None
    
    if cfg.quantized_layers is None:
        return None
    for pattern in get_key_patterns(name):
        if pattern in cfg.quantized_layers:
            return cfg.quantized_layers[pattern]
    return layer_quant_cfg


def check_exclude_layers(cfg, name, layer):
    if cfg.exclude_layers is None:
        return False
    return any(pattern in cfg.exclude_layers for pattern in get_key_patterns(name))


def modify_graph(model, modified_layers):
    for name, layer in modified_layers:
        submodules = name.split('.')[:-1]
        layer_name = name.split('.')[-1]
        setattr(model.get_submodule('.'.join(submodules)), layer_name, layer)


# 读取配置文件，获取量化配置和权重
def get_cfg_and_weights(quant_des_path):
    quant_des_path, filename, rank = replace_rank_suffix(quant_des_path)
    quant_algo_str = "quant_algo"
    with file_utils.safe_open(quant_des_path, "r", encoding="utf-8",
                              permission_mode=file_utils.CONFIG_FILE_PERMISSION) as reader:
        data = reader.read()
    quant_des_dict = json.loads(data, strict=False)
    logger.info(f"Quant Description Filename:{filename}")

    if not quant_des_dict:
        raise ParametersInvalid(f"quant_des_dict is none!")
    exclude_layers = [k for k, v in quant_des_dict.items() if v == "FLOAT"]
    valid_values = {item.value for item in QuantAlgorithm}  # 预计算有效值集合
    quantized_layers = {
        k: {quant_algo_str: QuantAlgorithm(v.upper())}
        for k, v in quant_des_dict.items()
        if isinstance(v, str) and v.upper() in valid_values
    }
    quant_algo = quant_des_dict.get("model_quant_type", None)
    if quant_algo is None:
        raise ParametersInvalid(f"quant_algo must be the type of QuantAlgorithm.")

    quant_config = {"quant_algo": quant_algo}
    quant_config.update({'exclude_layers': tuple(exclude_layers)})
    quant_config.update({'quantized_layers': quantized_layers})
    quant_config.update({quant_algo_str: QuantAlgorithm(quant_algo)})
    if isinstance(quant_config, dict):
        cfg = LayerQuantConfig.parse_from_dict(quant_config)
    else:
        cfg = quant_config

    quant_weight_dir = os.path.dirname(quant_des_path)
    if rank != -1:
        weight_name = f'quant_model_weight_{quant_algo.lower()}_{rank}.safetensors'
    else:
        weight_name = f'quant_model_weight_{quant_algo.lower()}.safetensors'
    quant_weight_path = os.path.join(quant_weight_dir, weight_name)
    quant_weight_path = file_utils.standardize_path(quant_weight_path)
    file_utils.check_file_safety(quant_weight_path,
        permission_mode=file_utils.MODELDATA_FILE_PERMISSION, max_file_size=MAX_WEIGHT_SIZE)
    quant_weights = safetensors.safe_open(quant_weight_path, framework="pytorch")
    logger.info(f"Quant Weight Path:{quant_weight_path}")

    return cfg, quant_weights


def validate_quantize_params(func):
    @wraps(func)
    def wrapper(model: nn.Module, quant_des_path: str, **kwargs):
        # 检查 model 类型
        if not isinstance(model, nn.Module):
            raise ParametersInvalid(f"The model must be the type of nn.Module, but currently got {type(model)}.")

        # 检查 quant_des_path 路径有效性
        if not isinstance(quant_des_path, str) or not quant_des_path.strip():
            raise ConfigError("Invalid string path for quant_des_path.")
        quant_des_path = file_utils.standardize_path(quant_des_path)
        file_utils.check_file_safety(quant_des_path, permission_mode=file_utils.MODELDATA_FILE_PERMISSION)

        timestep_config = kwargs.get('timestep_config')
        if timestep_config is not None and not isinstance(timestep_config, TimestepPolicyConfig):
            raise ParametersInvalid(f"Timestep_config must be the type of TimestepPolicyConfig,"
                "but currently got {type(timestep_config)}.")

        dtype = kwargs.get('dtype', torch.bfloat16)
        if not isinstance(dtype, torch.dtype) or dtype not in (torch.float16, torch.bfloat16):
            raise ParametersInvalid(f"Dtype must be torch.float16 or torch.bfloat16, but currently got {type(dtype)}.")

        module_map = kwargs.get('map', None)
        if module_map is not None:
            if not isinstance(module_map, Dict) or \
                    not all(isinstance(v, nn.Module) for v in module_map.values()) or \
                    not all(isinstance(k, nn.Module) for k in module_map.keys()):
                raise ParametersInvalid("The data type of map must be dictionary, and its KVType must be nn.Module.")

        return func(model, quant_des_path, **kwargs)

    return wrapper


# kwargs = cfg自定义配置， map自定义匹配规则
@validate_quantize_params
def quantize(model, quant_des_path, **kwargs):
    r"""
    The method is used to quant model.

    Args:
        model: Floating point models that need to be quantized.
        quant_des_path: The absolute path of the quantized weight descripter exported by modelslim.
        **kwargs:
            timestep_config: When using timetstep quantization, TimestepPolicyConfig needs to be passed in.
            dtype: Dtype specifies the type of the inverse quantization.
    Returns:
        Quantntifild Model.
    """
    cfg, quant_weights = get_cfg_and_weights(quant_des_path)

    if not isinstance(cfg, QuantConfig):
        logger.debug("cfg is not QuantConfig, Without enabling quantization.")
        return model

    if not cfg.layer_quantization_mode:
        logger.debug("Quantization content is none, Without enabling quantization.")
        return model

    modified_layers = []
    rank = int(os.getenv("RANK", 0))

    for name, layer in model.named_modules():
        # 跳过回退层
        if check_exclude_layers(cfg, name, layer):
            logger.debug("Skipping layer %s due to excluded configuration.", name)
            continue
        # 如果模型显式指定了融合层，以融合层指定的算法为最高优先级配置，否则从config里读取配置
        layer_quant_cfg = get_layer_quant_cfg(cfg, name, layer)
        if layer_quant_cfg is None:
            logger.debug("Cannot find the quantization configuration corresponding to %s.", name)
            continue

        # 以用户申明的融合算法为第一优先级，其次是读取配置中的
        layer_quant_mode = get_layer_quant_mode(name, layer, cfg)
        if layer_quant_mode is None:
            logger.debug("Cannot find the quantization mode corresponding to %s.", name)
            continue

        # 根据算法的要素dispatch到不同分支
        if layer_quant_mode.contains_activation_and_weight_quant():
            quant_layer, is_modified = smooth_quantize(name, layer, layer_quant_cfg, quant_weights, **kwargs)
            if is_modified:
                logger.debug(f"W8A8 Quant layer name:%s, Quant class name:%s.", name, quant_layer.__class__.__name__)
                modified_layers.append((name, quant_layer))
        elif layer_quant_mode.check_weight_only_mode():
            quant_layer, is_modified = weight_quantize(name, layer, layer_quant_cfg, quant_weights, **kwargs)
            if is_modified:
                logger.debug(f"Weight Quant layer name:%s, Quant class name:%s.", name, quant_layer.__class__.__name__)
                modified_layers.append((name, quant_layer))
        elif layer_quant_mode.contains_fa_quantization():
            add_fa_quant(layer, layer_quant_cfg, name, quant_weights)
            if rank == 0:
                logger.info(f"FA Quant layer name:%s, Quant class name:%s, Quant algo:%s.",
                            name, layer.__class__.__name__, layer_quant_cfg.quant_algo)

    # 执行改图
    modify_graph(model, modified_layers)
    torch.npu.empty_cache()

    return model