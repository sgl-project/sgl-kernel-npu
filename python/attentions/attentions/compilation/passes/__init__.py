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

import threading

once_flag = threading.Event()


def activate_pattern_once():

    def activate_pattern():
        from ..compiliation_config import CompilationConfig
        from .register_pattern_to_pass import register_pattern_to_pass
        import importlib

        pattern_registry = {
            "enable_rms_norm": ("RMSNormPatternGroup", "..patterns"),
            "enable_rope": ("RopePatternGroup", "..patterns"),
            "enable_adalayernorm": ("AdaLayerNormPatternGroup", "..patterns"),
            "enable_fast_gelu": ("GELUPatternGroup", "..patterns")
        }

        fusion_config = CompilationConfig.fusion_patterns
        for config_key, (pattern_group_name, pattern_module) in pattern_registry.items():
            if getattr(fusion_config, config_key, False):
                patterns_module = importlib.import_module(pattern_module, package=__package__)
                pattern_group = getattr(patterns_module, pattern_group_name)
                register_pattern_to_pass(pattern_group)

    if not once_flag.is_set():
        with threading.Lock():
            if not once_flag.is_set():
                activate_pattern()
                once_flag.set()