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

import torch
import torch.nn as nn
import torch_npu
from mindiesd.utils.exception import ParametersInvalid


def handle_profile_task(
        instruction, 
        upload_queue, 
        expert_load_collector_list, 
        dispatcher_list, 
        transfer_stream
    ):
    moe_layer_idx = instruction.moe_layer_idx
    if moe_layer_idx is not None and moe_layer_idx < len(expert_load_collector_list):
        expert_load_collector = expert_load_collector_list[moe_layer_idx]
        dispatcher = dispatcher_list[moe_layer_idx]
        if expert_load_collector:
            expert_load = expert_load_collector.get_expert_load().numpy()
            with dispatcher.update_lock:
                local_expert_list = dispatcher.local_expert_list
            response_data = {
                'moe_layer_idx': moe_layer_idx, 
                'load': expert_load,
                'local_expert_list': local_expert_list
            }
            upload_queue.put(response_data)


def handle_update_layout_task(
        instruction, 
        upload_queue, 
        expert_load_collector_list, 
        dispatcher_list, 
        transfer_stream
    ):
    moe_layer_idx = instruction.moe_layer_idx
    if moe_layer_idx is not None and moe_layer_idx < len(dispatcher_list):

        dispatcher = dispatcher_list[moe_layer_idx]
        if dispatcher.update_valid_tensor[0] == 1:
            return
        device = dispatcher.update_valid_tensor.device
        data_instruction = instruction.data

        device_indices = data_instruction['device_indices']
        local_expert_indices = data_instruction['local_expert_indices']
        local_expert_list = data_instruction['local_expert_list']
        expert_trans_tensor = data_instruction['expert_trans_tensor']

        device_indices_map_cpu = torch.tensor(device_indices)
        local_expert_indices_map_cpu = torch.tensor(local_expert_indices)

        weight1_local_list_cpu = [dispatcher.weight1_list_cpu[i] for i in local_expert_list]
        weight2_local_list_cpu = [dispatcher.weight2_list_cpu[i] for i in local_expert_list]
        
        with torch_npu.npu.stream(transfer_stream):

            device_indices_map = device_indices_map_cpu.to(device)
            local_expert_indices_map = local_expert_indices_map_cpu.to(device)

            # 直接修改data
            weight1_npu_list = [t.to(device, non_blocking=True) for t in weight1_local_list_cpu]
            weight2_npu_list = [t.to(device, non_blocking=True) for t in weight2_local_list_cpu]
            
            weight1 = nn.ParameterList(weight1_npu_list)
            weight2 = nn.ParameterList(weight2_npu_list)

            dispatcher.copy_module_weight_and_map(
                weight1=weight1, 
                weight2=weight2, 
                device_indices_map=device_indices_map, 
                local_expert_indices_map=local_expert_indices_map, 
                local_expert_list=local_expert_list, 
                expert_trans_tensor=expert_trans_tensor, 
                layer_idx=moe_layer_idx
            )


def handle_unknown_task(
        instruction, 
        upload_queue, 
        expert_load_collector_list, 
        dispatcher_list, 
        transfer_stream
    ):
    raise ParametersInvalid(f"Unknown task type: {instruction.task_type}")
