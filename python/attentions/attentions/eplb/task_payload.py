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

from enum import Enum, auto
from dataclasses import dataclass, field
from mindiesd.utils.exception import ParametersInvalid


class TaskType(Enum):
    PROFILE = auto()
    UPDATE_LAYOUT = auto()


@dataclass
class TaskPayload:
    task_type: TaskType

    worker_rank: int | None = None
    moe_layer_idx: int | None = None
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.task_type, TaskType):
            raise ParametersInvalid("task_type must be an instance of the TaskType Enum.")
        if not isinstance(self.data, dict):
            raise ParametersInvalid("data must be a dictionary.")
        if self.task_type == TaskType.UPDATE_LAYOUT:
            if not self.data:
                raise ParametersInvalid("data dictionary cannot be empty when task_type is UPDATE_LAYOUT.")
            if self.moe_layer_idx is None:
                raise ParametersInvalid("moe_layer_idx must be specified when task_type is UPDATE_LAYOUT.")