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
import torch_npu

from mindiesd.utils.logs.logging import logger
from .eplb_scheduler import get_manager_client
from .task_payload import TaskType, TaskPayload
from .task_transfer import ProfileTaskTransfer
from .task_handler import handle_profile_task, handle_update_layout_task, handle_unknown_task

TASK_DISPATCHER = {
    TaskType.PROFILE: handle_profile_task,
    TaskType.UPDATE_LAYOUT: handle_update_layout_task,
}


def parse_module(module):
    dispatcher_list = []
    expert_load_collector_list = []
    for _, child in module.named_modules():
        if hasattr(child, 'dispatcher') and hasattr(child, 'expert_load_collector'):
            dispatcher_list.append(child.dispatcher)
            expert_load_collector_list.append(child.expert_load_collector)
    return dispatcher_list, expert_load_collector_list


def expert_info_transfer_pool(
        module, 
        instruction_queue, 
        upload_queue, 
        device
    ):
    dispatcher_list, expert_load_collector_list = parse_module(module)
    transfer_stream = torch_npu.npu.Stream(device)

    for idx, collector in enumerate(expert_load_collector_list):
        collector.task_transfer = ProfileTaskTransfer(
            instruction_queue,
            idx,
            collector.lb_interval
        )

    while True:
        instruction = instruction_queue.get()
        if instruction is None or instruction == 'exit':
            logger.info(f"[ExpertInfoTransferPool] Get exit instruction")
            break
        if isinstance(instruction, TaskPayload):
            handler_function = TASK_DISPATCHER.get(instruction.task_type, handle_unknown_task)
            handler_function(instruction, upload_queue, expert_load_collector_list, dispatcher_list, transfer_stream)
        else:
            logger.debug(f"Unknown instruction: {instruction}")


def connect_to_schedule_manager(
        rank_in_group, 
        ip, 
        port,
        auth_key
    ):
    addr = (ip, port)
    manager = get_manager_client(addr, auth_key)
    manager.connect()
    logger.info(f"Connected to schedule manager, rank_in_group: {rank_in_group}")
    instruction_queue = manager.get_instruction_queues(rank=rank_in_group)
    upload_queue = manager.get_upload_queues(rank=rank_in_group)
    return instruction_queue, upload_queue


def construct_expert_info_transfer_pool(**kwargs):
    module = kwargs['module']
    rank_in_group = kwargs['rank_in_group']
    device = kwargs['device']
    ip = kwargs['ip']
    port = kwargs['port']
    auth_key = kwargs['auth_key']
    instruction_queue, upload_queue = connect_to_schedule_manager(rank_in_group, ip, port, auth_key)
    if instruction_queue is None or upload_queue is None:
        return None, None
    worker = threading.Thread(
        target=expert_info_transfer_pool, 
        args=(module, instruction_queue, upload_queue, device), 
        daemon=True
    )
    worker.start()
    return worker, instruction_queue