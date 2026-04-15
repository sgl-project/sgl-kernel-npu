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
from pathlib import Path
from functools import wraps
import os
from typing import Dict, Callable
import torch
from torch.library import Library
from ..utils import file_utils, ParametersInvalid, is_npu_available


MINDIE_NS = "attentions"  # 固定命名空间，与 torch.ops.attentions 对应


def _load_attentions_ops_library() -> None:
    """Load the MindIE custom operator shared library.
    
    Raises:
        ParametersInvalid: If the parent directory level is insufficient.
        FileNotFoundError: If the operator SO file is not found.
        PermissionError: If the SO file has invalid permissions.
    """
    current_path = Path(__file__).resolve()
    if len(current_path.parents) < 2:
        raise ParametersInvalid("Insufficient parent directory levels to locate plugin folder.")
    
    ops_path = current_path.parents[1] / "plugin"
    ops_path = file_utils.standardize_path(str(ops_path))
    ops_file = os.path.join(ops_path, "libPTAExtensionOPS.so")
    
    file_utils.check_file_safety(
        ops_file,
        permission_mode=file_utils.BINARY_FILE_PERMISSION
    )
    torch.ops.load_library(ops_file)


if is_npu_available():
    _load_attentions_ops_library()


def check_attentions_operator_exists(op_name: str) -> bool:
    """Check if a MindIE operator is registered in PyTorch.
    
    Args:
        op_name: Full name of the operator (e.g. "rope", "la")
    
    Returns:
        True if the operator exists, False otherwise.
    """
    try:
        getattr(torch.ops.attentions, op_name)
        return True
    except AttributeError:
        return False


if torch.__version__.startswith("0.1"):
    # PyTorch 2.1 使用 Library.impl
    _lib = Library(MINDIE_NS, "IMPL")
    
    def _compatible_register_fake(op_name: str):
        """Compatibility wrapper for PyTorch 2.1 fake registration."""
        def decorator(fake_func: Callable):
            @wraps(fake_func)
            def wrapper(*args, **kwargs):
                # Ensure all tensor inputs are on Meta device (required for PyTorch 2.1)
                args = [
                    a.to(device="meta") if isinstance(a, torch.Tensor) else a
                    for a in args
                ]
                kwargs = {
                    k: v.to(device="meta") if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
                return fake_func(*args, **kwargs)
            
            _lib.impl(op_name, wrapper, "Meta")
            return fake_func
        return decorator
else:
    # PyTorch 2.2+ 使用 register_fake 或 impl_abstract
    try:
        from torch.library import register_fake as _native_register_fake
    except ImportError:
        from torch.library import impl_abstract as _native_register_fake
    
    def _compatible_register_fake(op_name: str):
        """Compatibility wrapper for PyTorch 2.2+ fake registration."""
        return _native_register_fake(op_name)



def register_attentions_fake_op(op_name: str):
    """Decorator to register a fake implementation for a MindIE operator.
    
    Usage:
        @register_mindie_fake_op("rope")
        def rope_fake(x, cos, sin, mode):
            ...
    
    Args:
        op_name: Full name of the operator (e.g. "rope", "la")
    
    Returns:
        Decorator function that registers the fake implementation.
    """
    if not is_npu_available():
        def dummy_decorator(func):
            return func
        return dummy_decorator
    
    if not check_attentions_operator_exists(op_name):
        raise RuntimeError(
            f"MindIE operator {MINDIE_NS}::{op_name} not found! "
            "Ensure the SO library is loaded and the operator is registered with TORCH_LIBRARY."
        )
    
    return _compatible_register_fake(f"{MINDIE_NS}::{op_name}")