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
import torch.distributed as dist
from torch import nn


class DispatcherBase(nn.Module):

    def __init__(self, routed_expert_num, weight1, weight2, ep_rank, ep_size):
        super().__init__()
        self.routed_expert_num = routed_expert_num  # 路由专家数量
        self.local_expert_num = routed_expert_num // ep_size

        self.weight1_list = list(weight1.unbind(0))
        self.weight2_list = list(weight2.unbind(0))

        self.register_buffer('device_indices_map', torch.arange(self.routed_expert_num, dtype=torch.int32))
        self.register_buffer('local_expert_indices_map', torch.arange(self.routed_expert_num, dtype=torch.int32))

        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.init_expert_map() 

    def init_expert_map(self):
        device_indices_map = torch.arange(
            self.routed_expert_num, dtype=torch.int32) // (self.routed_expert_num // self.ep_size
        )
        local_expert_indices_map = torch.arange(
            self.routed_expert_num, dtype=torch.int32) - (self.routed_expert_num // self.ep_size
        ) * self.ep_rank
        self.set_device_and_local_expert_map(device_indices_map, local_expert_indices_map)

    def set_device_and_local_expert_map(self, device_indices_map: torch.Tensor, local_expert_indices_map: torch.Tensor):
        self.set_device_map(device_indices_map)
        self.set_local_expert_map(local_expert_indices_map)

    def set_device_map(self, device_indices_map: torch.Tensor):
        self.device_indices_map.copy_(device_indices_map)
    
    def set_local_expert_map(self, local_expert_indices_map: torch.Tensor):
        self.local_expert_indices_map.copy_(local_expert_indices_map)

    def get_device_indices(self, indices_expert: torch.Tensor):
        return self.device_indices_map[indices_expert]
        
    def get_local_expert_indices(self, indices_expert: torch.Tensor):
        return self.local_expert_indices_map[indices_expert]


class DynamicDispatcher(DispatcherBase):
    def __init__(self, routed_expert_num, weight1, weight2, ep_rank, ep_size):
        super().__init__(routed_expert_num, weight1, weight2, ep_rank, ep_size)
        
        self.register_buffer('update_valid_tensor', torch.zeros(1, dtype=torch.int32))

        self.update_flag = False
        self.update_lock = threading.Lock()
        self.module_state = {
            'weight1': None,
            'weight2': None,
        }

        self.weight1_list_cpu = [t.pin_memory() for t in self.weight1_list]
        self.weight2_list_cpu = [t.pin_memory() for t in self.weight2_list]

        self.local_expert_list = [self.local_expert_num * ep_rank + i for i in range(self.local_expert_num)] 
        self.expert_trans_tensor = None
    
    def get_expert_trans_tensor(self):
        return self.expert_trans_tensor
    
    def check_consistency(self):
        check_tensor = self.update_valid_tensor.clone()
        dist.all_reduce(check_tensor, op=dist.ReduceOp.MIN)
        self.update_flag = (check_tensor.item() == 1)
        return self.update_flag

    def copy_module_weight_and_map(self, **kwargs):
        weight1 = kwargs['weight1']
        weight2 = kwargs['weight2']
        device_indices_map = kwargs['device_indices_map']
        local_expert_indices_map = kwargs['local_expert_indices_map']
        local_expert_list = kwargs['local_expert_list']
        expert_trans_tensor = kwargs['expert_trans_tensor']
        layer_idx = kwargs['layer_idx']
        
        with self.update_lock:
            self.module_state = {
                'weight1': weight1,
                'weight2': weight2,
                'device_indices_map': device_indices_map,
                'local_expert_indices_map': local_expert_indices_map,
                'local_expert_list': local_expert_list,
                'layer_idx': layer_idx,
            }
            self.expert_trans_tensor = expert_trans_tensor
            self.update_valid_tensor.fill_(1)

    def update_module_weight_and_map(self):
        with self.update_lock:
            self.update_valid_tensor.fill_(0)
            self.device_indices_map.copy_(self.module_state['device_indices_map'])
            self.local_expert_indices_map.copy_(self.module_state['local_expert_indices_map'])
            self.local_expert_list = self.module_state['local_expert_list']
            self.local_expert_num = len(self.module_state['weight1'])
            self.update_flag = False
        result = (
            self.module_state['weight1'], 
            self.module_state['weight2'], 
            self.local_expert_num, 
            self.device_indices_map, 
            self.local_expert_indices_map, 
            self.local_expert_list
        )
        return result