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

import random
from dataclasses import dataclass
import numpy as np
import torch
from mindiesd.utils.logs.logging import logger
from mindiesd.utils.exception import ParametersInvalid


@dataclass
class LoadData:
    placement: np.ndarray
    shared_expert_id: int
    total_traffic: np.ndarray
    used_mems: dict
    global_expert_load: np.ndarray = None
    device_to_expert: dict = None
    sorted_experts: np.ndarray = None
    origin_device_to_expert: dict = None
    expert_trans_tensor: torch.Tensor = None


class EPLBService():
    """
    EPLB算法的基类

    Attributes:
        num_devices (int): 设备（NPU卡）的总数 (K)
        num_experts (int): MoE层中的专家数量 (N)
        expert_mems (dict): 每个专家的内存占用 (M_j)
        device_mems (dict): 每个设备的可用内存 (Mem_i)
        cost_local (float): 单个Token的本地计算成本 (C_comp)
        cost_remote (float): 单个Token的远程通信成本 (C_comm)
    """
    def __init__(self, num_devices, num_experts, expert_mems, device_mems, cost_local, cost_remote,
                 max_move_number, load_balance_threshold):
        self.num_devices = num_devices
        self.num_experts = num_experts
        self.expert_mems = expert_mems
        self.device_mems = device_mems
        self.cost_local = cost_local
        self.cost_remote = cost_remote
        self.max_move_number = max_move_number
        self.load_balance_threshold = load_balance_threshold

    @staticmethod
    def get_expert_total_demand(total_traffic):
        return np.sum(total_traffic, axis=0)
    
    def placement_greedy(self, traffic_data, origin_device_to_expert: dict = None, shared_expert_id: int = None):
        """
        基于贪心算法求解的专家放置策略

        Args:
            traffic_data (dict): 每张卡上处理的token路由到各个专家的数量
            origin_device_to_expert (dict): 每张卡上部署的专家列表
            shared_expert_id (dict): 共享专家 id
        Returns:
            bool: 是否触发更新
        """
        # --- 1. 数据预处理 ---
        # placement 初始化设备-专家的放置矩阵
        # sorted_experts 热度从高到低排序的专家列表
        # total_traffic 设备i的Token需要访问专家j的总次数
        # used_mems 初始化每个设备已使用的内存
        # global_expert_load
        placement, sorted_experts, total_traffic, used_mems, global_expert_load = self.data_preprocess(traffic_data)
        # 初始化专家交换映射矩阵：单位矩阵
        if origin_device_to_expert is not None:
            expert_trans_tensor_size = 0
            for _, value in origin_device_to_expert.items():
                expert_trans_tensor_size += len(value)
            expert_trans_tensor = torch.eye(expert_trans_tensor_size)
        else:
            expert_trans_tensor = torch.eye(self.num_experts)

        # --- 2. 处理共享专家 ---
        self.process_share_expert(placement, shared_expert_id, used_mems)
        # --- 3. 阶段1：满足约束的初始分配 ---
        initial_load_data = LoadData(placement=placement, shared_expert_id=shared_expert_id,
                                      total_traffic=total_traffic, used_mems=used_mems,
                                      origin_device_to_expert=origin_device_to_expert,
                                     sorted_experts=sorted_experts, expert_trans_tensor=expert_trans_tensor)
        device_to_expert = self.initial_placement(initial_load_data)
        # --- 4. 阶段2：迭代复制优化minmax目标 ---
        optimize_load_data = LoadData(placement=placement, shared_expert_id=shared_expert_id,
                                      total_traffic=total_traffic, used_mems=used_mems,
                                      global_expert_load=global_expert_load, device_to_expert=device_to_expert)
        self.optimize_min_max(optimize_load_data)

        return {
            "final_placement": placement,
            "final_memory_usage": used_mems,
            "device_to_expert_map": device_to_expert,
            "expert_trans_tensor": expert_trans_tensor
        }

    def optimize_min_max(
            self,
            load_data: LoadData
        ):
        pass

    def initial_placement(self, load_data: LoadData):
        logger.debug("--- [Phase 1] Start initial placement ---")
        device_to_expert = {i: [] for i in range(self.num_devices)}
        # 从待放置列表中移除已被强制部署的共享专家
        if load_data.shared_expert_id is not None:
            load_data.sorted_experts = [exp for exp in load_data.sorted_experts if exp != load_data.shared_expert_id]
        for expert_id in load_data.sorted_experts:
            best_device = -1
            min_estimated_load = float('inf')

            # 遍历所有设备，为当前专家寻找初始位置
            for device_id in range(self.num_devices):
                # 检查内存约束：当前设备是否能容纳这个专家
                if load_data.used_mems[device_id] + self.expert_mems[expert_id] <= self.device_mems[device_id]:
                    # 寻找预估负载最小的设备，用“已用内存”作为负载代理
                    # 选择一个较为空闲的设备，实现初始布局的均衡。
                    if load_data.used_mems[device_id] < min_estimated_load:
                        min_estimated_load = load_data.used_mems[device_id]
                        best_device = device_id

            # 如果找到了一个可以放置的位置
            if best_device != -1:
                load_data.placement[best_device, expert_id] = 1
                device_to_expert[best_device].append(expert_id)
                load_data.used_mems[best_device] += self.expert_mems[expert_id]
                logger.debug(
                    f"  -> Place high-load expert {expert_id} to device {best_device} "
                    f"(Current memory: {load_data.used_mems[best_device]:.1f}GB)")
            else:
                # 报错：如果每个专家连一个初始位置都找不到，说明无解
                raise MemoryError(
                    f"Error：expert {expert_id} (need {self.expert_mems[expert_id]}GB) "
                    f"Unable to locate initial position for this expert on any device, check memory configuration."
                )
        logger.debug("--- [Phase 1] Initial placement completed. ---\n")

        return device_to_expert

    def data_preprocess(self, traffic_data):
        total_traffic = np.zeros((self.num_devices, self.num_experts))
        for device_id, expert_requests in traffic_data.items():
            total_traffic[device_id] += expert_requests
        # 初始化部署矩阵 X_ij, 初始时所有专家都未放置
        placement = np.zeros((self.num_devices, self.num_experts), dtype=int)
        # 记录每个设备已使用的内存
        used_mems = {i: 0.0 for i in range(self.num_devices)}
        # 每个专家的总需求量
        expert_total_demand = self.get_expert_total_demand(total_traffic)
        # 按需求量从大到小对专家进行排序
        sorted_experts = np.argsort(expert_total_demand)[::-1]
        return placement, sorted_experts, total_traffic, used_mems, expert_total_demand

    def process_share_expert(self, placement, shared_expert_id, used_mems):
        if shared_expert_id is not None:
            logger.debug(f"--- [Preprocessing] Shared Expert {shared_expert_id} detected, "
                         f"forcing deployment on all devices ---")
            shared_mem = self.expert_mems[shared_expert_id]
            for i in range(self.num_devices):
                # 检查内存是否足够
                if self.device_mems[i] < shared_mem:
                    raise MemoryError(
                        f"Device {i} (memory {self.device_mems[i]}GB) cannot accommodate the shared expert."
                        f"{shared_expert_id} (need {shared_mem}GB)。"
                    )
                placement[i, shared_expert_id] = 1
                used_mems[i] += shared_mem
            logger.debug("--- [Preprocessing] Shared expert deployment completed ---\n")


class A2ARedundantExpertService(EPLBService):
    """
    面向All-to-all通信方式下的冗余专家动态调度
    """
    def __init__(self, num_devices, num_experts, expert_mems, device_mems, cost_local, cost_remote,
                max_move_number, load_balance_threshold):
        super().__init__(
            num_devices, num_experts, expert_mems, device_mems, cost_local, cost_remote,
            max_move_number, load_balance_threshold)

    def optimize_min_max(
            self,
            load_data: LoadData
        ):
        logger.debug("--- [Phase 2] Starting iterative replication optimization.---")
        iteration = 1
        while True:
            # 在每一轮迭代中，找到全局“性价比”最高的复制操作。
            best_move = None
            max_score = 1e-9  # 使用很小的正数，确保只选择有实际正收益的操作

            # 每次迭代都重新计算当前所有设备的负载，以找到当前的瓶颈
            current_loads = np.zeros(self.num_devices)
            for i in range(self.num_devices):
                load = 0
                for j in range(self.num_experts):
                    traffic = load_data.total_traffic[i, j]
                    if load_data.placement[i, j] == 1:
                        load += traffic * self.cost_local  # 本地计算成本
                    else:
                        load += traffic * self.cost_remote  # 远程通信成本
                current_loads[i] = load

            bottleneck_device = np.argmax(current_loads)
            max_load = np.max(current_loads)
            logger.debug(f"\nRound {iteration} | The current system bottleneck (maximum load): {max_load:,.0f} "
                         f"(on device {bottleneck_device})")

            # 遍历所有可能的“复制操作”
            # “复制操作”是指将专家 j 复制到设备 i，前提是 i 上没有 j，且内存足够。
            for expert_id in range(self.num_experts):

                # 跳过已被强制部署的共享专家
                if load_data.shared_expert_id is not None and expert_id == load_data.shared_expert_id:
                    continue

                for device_id in range(self.num_devices):

                    # 如果专家已经存在于此设备，或内存不足，则跳过
                    if load_data.placement[device_id, expert_id] == 1:
                        continue
                    if load_data.used_mems[device_id] + self.expert_mems[expert_id] > self.device_mems[device_id]:
                        continue

                    # --- 计算收益分数 ---
                    # 收益分数 = 收益 / 成本

                    # 收益 (Gain): "因在设备 i 复制专家 j 而节省的负载"
                    # 收益 = T * (cost_remote - cost_local)
                    traffic_from_device = load_data.total_traffic[device_id, expert_id]
                    gain = traffic_from_device * (self.cost_remote - self.cost_local)

                    # 成本 (Cost): "复制专家 j 所需的内存"
                    cost_mem = self.expert_mems[expert_id]

                    # 如果没有流量需求，复制没有意义
                    if gain <= 0:
                        continue

                    score = gain / cost_mem

                    if score > max_score:
                        max_score = score
                        best_move = (device_id, expert_id, gain)

            # --- 迭代停止 ---
            # 当找不到任何一个收益分数更大的机会时 (所有可能的专家复制，带来的内存开销都得不偿失)，或者内存不足而无法操作时，算法停止。
            if best_move is None:
                logger.debug("\n--- [Phase 2] Optimization completed: No more beneficial replication operations "
                             "found.---")
                break

            # 执行本轮找到的最佳移动
            dev_to_add, exp_to_add, move_gain = best_move
            load_data.placement[dev_to_add, exp_to_add] = 1
            load_data.device_to_expert[dev_to_add].append(exp_to_add)
            load_data.used_mems[dev_to_add] += self.expert_mems[exp_to_add]

            logger.debug(f"  -> Best Move: Copy Expert {exp_to_add} to Device {dev_to_add} "
                         f"(Benefit Score: {max_score:,.2f})")
            logger.debug(f"     Load benefit: Reduced load by {move_gain:,.0f} for Device {dev_to_add}")
            logger.debug(f"     New memory status: "
                         f"{load_data.used_mems[dev_to_add]:.1f}GB / {self.device_mems[dev_to_add]}GB")

            iteration += 1


class AGRedundantExpertService(EPLBService):
    """
    面向All-Gather通信方式下的冗余专家动态调度
    """
    def __init__(self, num_devices, num_experts, expert_mems, device_mems, cost_local, cost_remote,
                 max_move_number, load_balance_threshold):
        super().__init__(num_devices, num_experts, expert_mems, device_mems, cost_local, cost_remote,
                         max_move_number, load_balance_threshold)

    @staticmethod
    def get_expert_total_demand(total_traffic):
        return np.mean(total_traffic, axis=0)

    def optimize_min_max(self, load_data: LoadData):
        logger.debug("--- [Phase 2] Starting iterative replication optimization ---")
        iteration = 1
        while True:
            # --- 修改点 3: 重写负载计算逻辑 ---
            # 1. 计算当前每个专家的副本数 K_j
            replicas_per_expert = np.sum(load_data.placement, axis=0)
            # 放置一个极小值避免除以0，对于未放置的专家（理论上几乎不会发生）
            replicas_per_expert[replicas_per_expert == 0] = 1e-9

            # 2. 计算每个设备的计算负载
            current_loads = np.zeros(self.num_devices)
            for i in range(self.num_devices):
                load = 0
                for j in range(self.num_experts):
                    if load_data.placement[i, j] == 1:
                        # 负载 = (专家j的全局负载 / 专家j的副本数) * 计算成本
                        load += (load_data.global_expert_load[j] / replicas_per_expert[j]) * self.cost_local
                current_loads[i] = load

            bottleneck_device = np.argmax(current_loads)
            max_load = np.max(current_loads)
            logger.debug(f"\nRound {iteration} | The current system bottleneck (maximum load): {max_load:,.0f} "
                         f"(on device {bottleneck_device})")
            logger.debug(f" Devices load: {[f'{device_load:,.0f}' for device_load in current_loads]}")

            # --- 修改点 4: 重写收益评估逻辑 ---
            best_move = None
            max_score = 1e-9

            for expert_id in range(self.num_experts):
                if load_data.shared_expert_id is not None and expert_id == load_data.shared_expert_id:
                    continue

                for device_id in range(self.num_devices):
                    if load_data.placement[device_id, expert_id] == 1:
                        continue
                    if load_data.used_mems[device_id] + self.expert_mems[expert_id] > self.device_mems[device_id]:
                        continue

                    # 模拟移动
                    temp_placement = load_data.placement.copy()
                    temp_placement[device_id, expert_id] = 1

                    temp_replicas = np.sum(temp_placement, axis=0)
                    temp_replicas[temp_replicas == 0] = 1e-9

                    """ 核心的代码逻辑是为了计算出模拟移动后，整体的新负载
                    一次模拟的移动后，只有跟这个专家相关的设备有变化，无需全遍历 """
                    new_loads = current_loads.copy()
                    # 所产生的专家热度分流效果
                    device_add_load = (
                        load_data.global_expert_load[expert_id] / temp_replicas[expert_id]
                    ) * self.cost_local
                    # 新复制了专家的那个设备增加负载
                    new_loads[device_id] += device_add_load
                    # 放置了相同专家的其他设备分享分流效果
                    devices_for_expert = np.where(load_data.placement[:, expert_id] == 1)[0]
                    new_loads[devices_for_expert] -= device_add_load / (temp_replicas[expert_id] - 1)

                    new_max_load = np.max(new_loads)

                    # 收益 = 系统最大负载的降低值
                    gain = max_load - new_max_load
                    cost_mem = self.expert_mems[expert_id]

                    if gain <= 0:
                        continue

                    score = gain / cost_mem

                    if score > max_score:
                        max_score = score
                        best_move = (device_id, expert_id, gain)

            if best_move is None:
                logger.debug("\n--- [Phase 2] Optimization completed: No more beneficial replication operations "
                             "found.---")
                break

            dev_to_add, exp_to_add, move_gain = best_move
            load_data.placement[dev_to_add, exp_to_add] = 1
            load_data.device_to_expert[dev_to_add].append(exp_to_add)
            load_data.used_mems[dev_to_add] += self.expert_mems[exp_to_add]

            logger.debug(f"  -> Best Move: Copy Expert {exp_to_add} to Device {dev_to_add} "
                         f"(Benefit Score: {max_score:,.2f})")
            logger.debug(f"     Load benefit: System maximum load reduced by {move_gain:,.0f}")
            logger.debug(f"     New memory status: "
                         f"{load_data.used_mems[dev_to_add]:.1f}GB / {self.device_mems[dev_to_add]}GB")

            iteration += 1


class ExpertExchangeService(EPLBService):
    """
    基于专家交换的动态调度方案
    """
    def __init__(self, num_devices, num_experts, expert_mems, device_mems, cost_local, cost_remote,
                 max_move_number, load_balance_threshold):
        super().__init__(num_devices, num_experts, expert_mems, device_mems, cost_local, cost_remote,
                         max_move_number, load_balance_threshold)

    def initial_placement(self, load_data: LoadData):
        device_to_expert = load_data.origin_device_to_expert.copy()
        device_loads = [0] * self.num_devices
        expert_start_index = 0
        # 初始设备负载情况
        if device_to_expert is None:
            device_to_expert = {}
            for device_id in range(self.num_devices):
                expert_end_index = (device_id + 1) * (self.num_experts // self.num_devices)
                device_loads[device_id] = load_data.total_traffic[device_id, expert_start_index: expert_end_index].sum()
                device_to_expert[device_id] = [i for i in range(expert_start_index, expert_end_index)]
                for expert_id in range(expert_start_index, expert_end_index):
                    load_data.used_mems[device_id] += self.expert_mems[expert_id]
                expert_start_index = expert_end_index
        else:
            for device_id in range(self.num_devices):
                device_loads[device_id] = load_data.total_traffic[device_id, device_to_expert[device_id]].sum()
                for expert_id in device_to_expert[device_id]:
                    load_data.used_mems[device_id] += self.expert_mems[expert_id]

        logger.debug(f"------------ Initial load status of the devices: {device_loads} ---------------------")
        move_expert_cost = 0
        for move_num in range(self.max_move_number):
            logger.debug(f"------------ Round {move_num} ---------------------")
            # 计算最大最小负载的负载差
            max_load = max(device_loads)
            max_device_index = device_loads.index(max_load)
            min_load = min(device_loads)
            min_device_index = device_loads.index(min_load)
            delta_load = (max_load - min_load) // 2
            logger.debug(f"--- Max-min device load diff {delta_load * 2}")
            if delta_load * 2 < self.load_balance_threshold:
                logger.debug(f"------------ Max-min device load diff less than {self.load_balance_threshold} "
                             f"End the iteration ---------------------")
                break

            # 通过两个设备上的专家负载，计算专家间两两交换后带来的负载差与设备负载差的差值矩阵，形成一个二维矩阵，此时值最小的i,j，就是我们要交换的两个专家
            if max_device_index in device_to_expert and min_device_index in device_to_expert:
                expert_max_idx = device_to_expert[max_device_index]
                expert_min_idx = device_to_expert[min_device_index]
                max_load_device_traffic = load_data.total_traffic[max_device_index, expert_max_idx]
                min_load_device_traffic = load_data.total_traffic[min_device_index, expert_min_idx]
            else:
                raise ParametersInvalid(f"[greedy] expert not in index list")

            trans_traffic = np.abs((max_load_device_traffic[:, np.newaxis] - min_load_device_traffic) - delta_load)
            # 找到 trans_traffic 中最小值的全局索引
            flat_traffic = trans_traffic.flatten()
            sorted_indices = np.argsort(flat_traffic)
            # 遍历排序后的索引，找到第一个满足条件（交换的在目标设备上不能已存在）的交换对
            for idx in sorted_indices:
                # 将扁平索引转换为二维索引
                experid_from_max_to_min_index, experid_from_min_to_max_index = np.unravel_index(
                    idx, trans_traffic.shape
                )
                # 获取对应的专家ID
                experid_from_max_to_min = expert_max_idx[experid_from_max_to_min_index]
                experid_from_min_to_max = expert_min_idx[experid_from_min_to_max_index]

                # 检查条件：目标设备上不能已包含对方专家
                if (experid_from_min_to_max not in device_to_expert.get(max_device_index, []) and
                        experid_from_max_to_min not in device_to_expert.get(min_device_index, [])):
                    # 找到满足条件的交换对，跳出循环
                    break
            else:
                logger.debug("\n--- No more beneficial replication operations found.---")
                break

            move_expert_load = (load_data.total_traffic[max_device_index, experid_from_max_to_min]
                                - load_data.total_traffic[min_device_index, experid_from_min_to_max])

            # 计算模拟移动后的两个设备间的负载差，和之前的负载差相比，计算收益，收益大于0则执行移动
            new_delta_load = abs((max_load - move_expert_load) - (min_load + move_expert_load)) // 2
            gain = delta_load - new_delta_load
            if gain > 0:
                # 执行移动前进行防御性检查
                if (max_device_index in device_to_expert and
                        min_device_index in device_to_expert):
                    # 执行移动
                    device_to_expert[max_device_index][experid_from_max_to_min_index] = experid_from_min_to_max
                    device_to_expert[min_device_index][experid_from_min_to_max_index] = experid_from_max_to_min
                else:
                    raise IndexError("Device or expert index out of bounds")

                device_loads[max_device_index] -= move_expert_load
                device_loads[min_device_index] += move_expert_load
                logger.debug(
                    f"--- Move expert {experid_from_max_to_min} from device {max_device_index} to {min_device_index}, "
                    f"device {max_device_index} load "
                    f"reduced by {load_data.total_traffic[max_device_index, experid_from_max_to_min]}, "
                    f"device {min_device_index} load "
                    f"increased by {load_data.total_traffic[max_device_index, experid_from_max_to_min]}; "
                    f"Move expert {experid_from_min_to_max} from device {min_device_index} to {max_device_index},"
                    f"device {min_device_index} load "
                    f"reduced by {load_data.total_traffic[min_device_index, experid_from_min_to_max]}, "
                    f"device {max_device_index} load "
                    f"increased {load_data.total_traffic[min_device_index, experid_from_min_to_max]}. ")
                logger.debug(f"--- Latest Max-min device load diff: {new_delta_load * 2}")

                # 记录1、移动专家数，2、更新内存，3、更新专家交换映射矩阵
                move_expert_cost += 2
                load_data.used_mems[min_device_index] += self.expert_mems[experid_from_max_to_min]
                load_data.used_mems[min_device_index] -= self.expert_mems[experid_from_min_to_max]

                load_data.used_mems[max_device_index] += self.expert_mems[experid_from_min_to_max]
                load_data.used_mems[max_device_index] -= self.expert_mems[experid_from_max_to_min]

                i = experid_from_max_to_min_index + (max_device_index
                                                    * (load_data.expert_trans_tensor.shape[0] // self.num_devices))
                j = experid_from_min_to_max_index + (min_device_index
                                                    * (load_data.expert_trans_tensor.shape[0] // self.num_devices))
                load_data.expert_trans_tensor[:, [i, j]] = load_data.expert_trans_tensor[:, [j, i]]
                logger.debug(f"--- Current load status: {device_loads} ---")
        return device_to_expert


def process_final_placement(results, num_experts):
    current_placement = results["device_to_expert_map"]  # N * K
    revert_list = [[] for i in range(num_experts)]
    local_expert_indices = [[-1] * num_experts for i in range(len(current_placement))]
    for i, result_list in current_placement.items():
        for idx, expert in enumerate(result_list):
            revert_list[expert].append(i)
            local_expert_indices[i][expert] = idx
    device_indices = [[] for _ in range(len(current_placement))]

    for device_index, device_experts in current_placement.items():
        for expert in range(num_experts):
            if expert in device_experts:
                device_indices[device_index].append(device_index)
            else:
                device_indices[device_index].append(random.choice(revert_list[expert]))
    local_expert_list = []

    for _, val in current_placement.items():
        local_expert_list.append(val)

    expert_trans_tensor = results["expert_trans_tensor"]
    return device_indices, local_expert_indices, local_expert_list, expert_trans_tensor


def process_expert_num(load_dic, num_experts):
    for idx, load in load_dic.items():
        load_dic[idx] = np.diff(load, prepend=0).astype(dtype=np.int64)

    return load_dic


def eplb_greedy(**kwargs):
    """
    算法入口

    Args:
        response (dict): 每张卡上处理的token路由到各个专家的数量
                          格式: {device_id: np.array([...])}
        device_to_expert (dict): 每张卡上部署的专家列表
                          格式: {device_id: np.array([...])}
        algorithm_type (str): 根据不同场景分为 []
        world_size (int): ep_size
        expert_num (int): 全局专家个数
        max_move (int): EX模式下最大移动专家数量
        redundant (int): 冗余专家个数
    Returns:
        bool: 是否触发更新
        dict: 设备token转发路由
        dict: 每个设备上专家的本地索引
        list: 每个设备上部署的最新专家列表
        tensor: 在EX(专家交换)场景下生成的专家交换张量，可与原专家分布张量做矩阵乘法后得到最新的专家分布
    """
    response = kwargs['response']
    algorithm_type = kwargs['algorithm_type']
    device_to_expert = kwargs['device_to_expert']
    world_size = kwargs['world_size']
    expert_num = kwargs['expert_num']
    max_move = kwargs['max_move']
    redundant = kwargs['redundant']

    load_balance_threshold = 100
    cost_local = 1
    cost_remote = 10
    current_pattern = process_expert_num(response, expert_num)

    update = True
    if (response[0] == 0).all():
        update = False

    expert_per_rank = (expert_num // world_size) + redundant

    # 2. 成本和资源约束
    expert_mems = {i: 1.0 for i in range(expert_num)}
    device_mems = {i: expert_per_rank for i in range(world_size)}

    # 定义每个场景关联的算法服务
    handlers = {
        'A2A': A2ARedundantExpertService(world_size, expert_num, expert_mems, device_mems, cost_local, cost_remote,
                                         max_move, load_balance_threshold),
        'AG': AGRedundantExpertService(world_size, expert_num, expert_mems, device_mems, cost_local, cost_remote,
                                       max_move, load_balance_threshold),
        'EX': ExpertExchangeService(world_size, expert_num, expert_mems, device_mems, cost_local, cost_remote,
                                       max_move, load_balance_threshold)
    }
    algorithm_service = handlers.get(algorithm_type)
    result = algorithm_service.placement_greedy(current_pattern, device_to_expert)
    output = process_final_placement(result, expert_num)
    device_indices, local_expert_indices, local_expert_list, expert_trans_tensor = output
    logger.info(f"current_placement:{local_expert_list}")
    return update, device_indices, local_expert_indices, local_expert_list, expert_trans_tensor