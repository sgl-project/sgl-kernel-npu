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

import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

import torch


class GraphModulePass(ABC):
    """Use the same interface as Inductor's CustomGraphPass"""

    @abstractmethod
    def __call__(self, graph: torch.fx.GraphModule) -> None:
        """
        Implementation of the custom pass.
        """

    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used for code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        hasher = hashlib.sha256()
        src = inspect.getsource(self.__class__)
        hasher.update(src.encode("utf-8"))
        return hasher.hexdigest()