# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2

"""Initialize the official provider and resolve its EQBSA dispatcher once."""

import importlib

import torch

from . import _backend

if _backend != "mindiesd":
    raise RuntimeError(
        "EQBSA uses official MindIE-SD. Start a new process with "
        "ATTENTIONS_BACKEND=mindiesd before importing attentions to avoid "
        "loading both native operator providers."
    )

try:
    # Official MindIE-SD owns OPP paths, the native plugin and Fake registrations.
    importlib.import_module("mindiesd")
except ModuleNotFoundError as exc:
    if exc.name != "mindiesd":
        raise
    raise RuntimeError(
        "EQBSA requires official MindIE-SD. Install a MindIE-SD wheel containing "
        "A5/950 EQBSA and matching your CANN, torch and torch_npu versions."
    ) from exc


def load_eqbsa():
    """Keep official operator and Fake registration; add no torch namespace."""
    try:
        return torch.ops.mindiesd.eagle_quant_block_sparse_attention.default
    except AttributeError as exc:
        raise RuntimeError(
            "The installed MindIE-SD does not register "
            "torch.ops.mindiesd.eagle_quant_block_sparse_attention. "
            "Install an official wheel containing A5/950 EQBSA."
        ) from exc
