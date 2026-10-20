#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
#
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
from pathlib import Path

__all__ = ["eagle_quant_block_sparse_attention"]

_native_library = Path(__file__).resolve().parent / "plugin" / "libPTAExtensionOPS.so"
_backend = os.environ.get(
    "ATTENTIONS_BACKEND", "native" if _native_library.is_file() else "mindiesd"
)
if _backend not in {"native", "mindiesd"}:
    raise ValueError("ATTENTIONS_BACKEND must be 'native' or 'mindiesd'.")

if _backend == "native":
    from .env import set_environment_variables

    set_environment_variables()

    from .lib import load_library

    load_library()
    # Preserve wildcard imports of the existing native initialization helpers.
    __all__ = ["set_environment_variables", "load_library"]


def __getattr__(name):
    if name == "eagle_quant_block_sparse_attention":
        from .eqbsa import eagle_quant_block_sparse_attention as function
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = function
    return function
