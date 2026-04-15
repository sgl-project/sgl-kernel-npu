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

import time
import atexit
from dataclasses import dataclass
import logging
from typing import Optional, Union

import zmq
import torch
import torch_npu
from .utils.exception import TorchError, ParametersInvalid

logger = logging.getLogger(__name__)


class ShareMemoryManager:
    def __init__(self, 
                 instance_world_size: int, 
                 instance_id: int,
                 master_addr: str = "127.0.0.1",
                 base_port: int = 5555):
        self.instance_world_size = instance_world_size
        self.instance_id = instance_id
        self.device_id = torch.npu.current_device()
        self.master_addr = master_addr
        self.base_port = base_port
        self.is_master = (instance_id == 0)
        
        self.pub_port = self.base_port + self.device_id + 100
        self.rep_port = self.pub_port + 1
        
        if self.is_master:
            self.rep_socket = ZMQ_CONTEXT.socket(zmq.REP)
            self.rep_socket.bind(f"tcp://{self.master_addr}:{self.rep_port}")
            self.rep_socket.setsockopt(zmq.RCVTIMEO, 10000)
            
            self.pub_socket = ZMQ_CONTEXT.socket(zmq.PUB)
            self.pub_socket.bind(f"tcp://{self.master_addr}:{self.pub_port}")
        else:
            self.sub_socket = ZMQ_CONTEXT.socket(zmq.SUB)
            self.sub_socket.connect(f"tcp://{self.master_addr}:{self.pub_port}")
            self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.sub_socket.setsockopt(zmq.RCVTIMEO, 5000)
            
            self.req_socket = ZMQ_CONTEXT.socket(zmq.REQ)
            self.req_socket.connect(f"tcp://{self.master_addr}:{self.rep_port}")

    @classmethod
    def cleanup(cls):
        global ZMQ_CONTEXT
        if ZMQ_CONTEXT:
            ZMQ_CONTEXT.term()
            ZMQ_CONTEXT = None

    def broadcast_handle(self, handle: int) -> int:
        if self.is_master:
            ready_count = 0
            while ready_count < self.instance_world_size - 1:
                try:
                    self.rep_socket.recv()
                    self.rep_socket.send(b"ACK")
                    ready_count += 1
                except zmq.Again as e:
                    raise TimeoutError(f"Master timeout waiting for child processes ready") from e
            
            logger.debug("Master broadcasting handle on tcp://%s:%s", self.master_addr, self.pub_port)
            self.pub_socket.send_pyobj(handle)
            logger.debug("Master broadcasted handle: %s", handle)
            return handle
        else:
            logger.debug(
                "Device %s subscribing to handle on tcp://%s:%s",
                self.device_id, self.master_addr, self.pub_port
            )
            self.req_socket.send(b"READY")
            self.req_socket.recv()
            
            try:
                recv_handle = self.sub_socket.recv_pyobj()
            except zmq.Again as e:
                raise TimeoutError(f"Device {self.device_id} timeout waiting for share handle") from e
            logger.debug("Device %s received handle: %s", self.device_id, recv_handle)
            return recv_handle


ZMQ_CONTEXT = zmq.Context.instance()
manager: Optional[ShareMemoryManager] = None


def init_share_memory(instance_world_size: int, 
                             instance_id: int,
                             master_addr: str = "127.0.0.1",
                             base_port: int = 5555) -> ShareMemoryManager:
    """
    设置共享内存管理器实例
    Args:
        instance_world_size: 总实例数
        instance_id: 当前实例ID
        master_addr: ZMQ通信主地址（默认127.0.0.1，支持跨机通信）
        base_port: ZMQ通信基础端口（默认5555，可自定义避免端口冲突）
    Returns:
        ShareMemoryManager: 单例管理器实例
    """
    global manager
    manager = ShareMemoryManager(instance_world_size, instance_id, master_addr, base_port)
    return manager


def get_share_memory_manager() -> ShareMemoryManager:
    global manager
    if not manager:
        raise ParametersInvalid("ShareMemoryManager has not been initialized."
            "Please call init_share_memory first.")
    return manager


def _check_device_and_dtype(module: torch.nn.Module, 
                            target_device: Optional[torch.device], 
                            target_dtype: Optional[torch.dtype]):
    cur_device = next(module.parameters()).device
    cpu_device = torch.device("cpu")
    meta_device = torch.device("meta")

    if cur_device == cpu_device and (target_device is None or target_device == cpu_device):
        return True, torch.nn.Module.to(module, target_device, target_dtype), None, None
    if cur_device == meta_device or target_device == meta_device:
        return True, torch.nn.Module.to(module, target_device, target_dtype), None, None
    
    device_id = torch.npu.current_device()
    npu_device = torch.device(f"npu:{device_id}")
    if cur_device == npu_device:
        return True, module, None, None
    
    if target_dtype is not None and not (target_dtype.is_floating_point or target_dtype.is_complex):
        raise ParametersInvalid(
            f'nn.Module.to only accepts floating point or complex dtypes, but got desired dtype={target_dtype}'
        )
    
    return False, None, npu_device, device_id


def share_memory(module: torch.nn.Module,
                    device: Optional[Union[str, torch.device]] = None,
                    dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """    
    Args:
        module (torch.nn.Module): 待迁移的模型实例（必传，强类型校验）
        device (Union[str, torch.device], optional): 目标设备，如"npu:0"/torch.device("npu:0")
        dtype (torch.dtype, optional): 目标数据类型，如torch.float16/torch.bfloat16
    
    Returns:
        torch.nn.Module: 共享内存迁移后的模型实例
    
    Raises:
        ParametersInvalid: 模型类型非法、数据类型非法、管理器未初始化
        TimeoutError: 子进程接收共享句柄超时
    """
    if not isinstance(module, torch.nn.Module):
        raise ParametersInvalid(f"第一个参数必须是torch.nn.Module实例，当前传入：{type(module)}")
    
    target_device = torch.device(device) if isinstance(device, str) else device
    target_dtype = dtype
    
    cur_device = next(module.parameters()).device
    logger.debug("%s from device '%s' to device '%s' and dtype '%s'",
                type(module).__name__, cur_device, target_device, target_dtype)
    
    should_fallback, fallback_result, _, _ = _check_device_and_dtype(module, target_device, target_dtype)
    if should_fallback:
        return fallback_result
    
    sm_manager = get_share_memory_manager()

    for _, param in list(module.named_parameters()) + list(module.named_buffers()):
        cast_dtype = target_dtype if target_dtype is not None else param.dtype

        if sm_manager.instance_id == 0:
            param.data = param.to(device=target_device, dtype=cast_dtype)
            local_storage = param.storage()._share_npu_()
            sm_manager.broadcast_handle(local_storage)
        else:
            new_tensor = torch.empty_like(param, device=target_device)
            recv_storage = sm_manager.broadcast_handle(None)
            rebuild_storage = torch.UntypedStorage._new_shared_npu(*recv_storage)
            new_tensor.set_(rebuild_storage)
            param.data = new_tensor.view(param.shape)

    torch.npu.empty_cache()
    return module


atexit.register(ShareMemoryManager.cleanup)