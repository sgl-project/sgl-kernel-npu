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

from dataclasses import dataclass
from functools import wraps
from collections import OrderedDict

import torch


attn_cache = OrderedDict()


def lru_cache_by_attn_param(maxsize=None):
    def decorator(func):
        
        @wraps(func)
        def wrapper(attn_param, *args, **kwargs):
            cache_key = attn_param.to_hash()
            if cache_key in attn_cache:
                attn_cache.move_to_end(cache_key)
                return attn_cache[cache_key]

            result = func(attn_param, *args, **kwargs)
            attn_cache[cache_key] = result

            if maxsize is not None and len(attn_cache) > maxsize:
                attn_cache.popitem(last=False)
            return result

        return wrapper
    return decorator


@dataclass
class AttentionParam:
    batch_size: int
    head_num: int
    head_dim: int
    q_seqlen: int
    kv_seqlen: int
    dtype: torch.dtype
    head_first: bool

    def to_str(self):
        param_str = f"batch_size:{self.batch_size}, head_num:{self.head_num}, head_dim:{self.head_dim}," \
            f" q_seqlen:{self.q_seqlen}, kv_seqlen:{self.kv_seqlen}, dtype:{self.dtype}, head_fist: {self.head_first}"
        return param_str

    def to_hash(self):
        return hash((
            self.batch_size, self.head_num, self.head_dim, self.q_seqlen, self.kv_seqlen, self.dtype, self.head_first))