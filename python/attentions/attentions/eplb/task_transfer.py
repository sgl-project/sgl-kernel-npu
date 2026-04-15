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

import queue

from mindiesd.utils.logs.logging import logger
from .task_payload import TaskType, TaskPayload


class ProfileTaskTransfer:
    def __init__(
            self, 
            task_queue: queue.Queue, 
            moe_layer_idx: int, 
            lb_interval: int = 1
        ):
        self.instruction_queue = task_queue
        self.moe_layer_idx = moe_layer_idx
        self.lb_interval = lb_interval
        self.flag = 0
    
    def profile_emit_task(self):
        task_payload = TaskPayload(
            task_type=TaskType.PROFILE, 
            moe_layer_idx=self.moe_layer_idx
        )
        if self.instruction_queue:
            self.flag += 1
            if self.flag != self.lb_interval:
                return
            self.flag = 0
            try:
                self.instruction_queue.put_nowait(task_payload)
            except queue.Full:
                logger.info(f"[Warning] instruction_queue full!!!")
                pass


class UpdateTaskTransfer:
    def __init__(
            self, 
            task_queue: queue.Queue, 
            moe_layer_idx
        ):
        self.instruction_queue = task_queue
        self.moe_layer_idx = moe_layer_idx

    def update_emit_task(
            self, 
            device_indices_list, 
            local_expert_indices_list, 
            local_expert_list, 
            expert_trans_tensor, 
            world_size
        ):
        for rank in range(world_size):
            layout_command = {
                'device_indices': device_indices_list[rank], 
                'local_expert_indices': local_expert_indices_list[rank], 
                'local_expert_list': local_expert_list[rank], 
                'expert_trans_tensor': expert_trans_tensor
            }
            task_payload = TaskPayload(
                task_type=TaskType.UPDATE_LAYOUT, 
                worker_rank=rank, 
                moe_layer_idx=self.moe_layer_idx, 
                data=layout_command
            )
            self.instruction_queue[rank].put(task_payload)