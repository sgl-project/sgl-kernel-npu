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
import dataclasses


@dataclasses.dataclass(frozen=False)
class FusionPatterns:
    enable_rms_norm: bool = True
    enable_rope: bool = True
    enable_adalayernorm: bool = True
    enable_fast_gelu: bool = True


class CompilationConfig:
    enable_freezing: bool = True
    graph_log_url: str | None = None
    fusion_patterns: FusionPatterns = FusionPatterns()

    def __init__(self):
        raise RuntimeError("CompilationConfig is a static class, do not instantiate it")