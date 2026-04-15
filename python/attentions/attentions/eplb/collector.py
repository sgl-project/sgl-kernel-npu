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

import threading
import torch
import torch_npu
from torch import nn


class ExpertLoadCollector(nn.Module):
    def __init__(self, routed_expert_num, lb_interval: int = 1) -> None:
        super().__init__()
        self.routed_expert_num = routed_expert_num
        self.register_buffer('expert_data_buffer', torch.zeros(self.routed_expert_num, dtype=torch.long))
        self.register_buffer('expert_group_list', torch.zeros(self.routed_expert_num, dtype=torch.long))
        self.experts_load_cpu = torch.zeros(self.routed_expert_num, dtype=torch.long).pin_memory()
        self.buffer_lock = threading.Lock()

        self.lb_interval = lb_interval
        self.task_transfer = None

    def get_expert_load(self):
        with self.buffer_lock:
            self.expert_data_buffer.copy_(self.expert_group_list)
            self.reset()
        self.experts_load_cpu.copy_(self.expert_data_buffer)
        return self.experts_load_cpu

    def collect_expert_load(self, indices_expert: torch.Tensor):
        expanded_buffer = torch_npu.npu_moe_compute_expert_tokens(indices_expert, self.routed_expert_num)
        with self.buffer_lock:
            self.expert_group_list.add_(expanded_buffer)
        
        if self.task_transfer:
            self.task_transfer.profile_emit_task()

    def reset(self):
        self.expert_group_list.zero_()