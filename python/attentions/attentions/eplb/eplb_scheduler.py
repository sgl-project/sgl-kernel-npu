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

import os
import multiprocessing
from multiprocessing.managers import BaseManager
import threading
import queue
import time
import random
import argparse

from mindiesd.utils.logs.logging import logger
from mindiesd.utils.exception import ModelExecError
from .task_transfer import UpdateTaskTransfer
from .greedy_algorithm import eplb_greedy

upload_queues = {}
instruction_queues = {}


class ScheduleManager(BaseManager):
    pass

ScheduleManager.register('get_upload_queues', callable=lambda rank: upload_queues[rank])
ScheduleManager.register('get_instruction_queues', callable=lambda rank: instruction_queues[rank])


def get_args():
    parser = argparse.ArgumentParser(description="EPLB scheduler")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--expert_num", type=int, default=32)
    parser.add_argument("--block_num", type=int, default=30)
    parser.add_argument("--max_move", type=int, default=5)
    parser.add_argument("--redundant", type=int, default=0)
    parser.add_argument("--mode", type=str, default="A2A")
    parser.add_argument("--auth_key", type=str, default=os.environ.get("EPLB_AUTH_KEY", "secret_key"))

    return parser.parse_args()


def start_manager_server(addr, auth_key):
    auth_bytes = auth_key.encode('utf-8')
    multiprocessing.current_process().authkey = auth_bytes
    manager = ScheduleManager(address=addr, authkey=auth_bytes)
    server = manager.get_server()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()


def get_manager_client(addr, auth_key):
    auth_bytes = auth_key.encode('utf-8')
    manager = ScheduleManager(address=addr, authkey=auth_bytes)
    return manager


def run_scheduler(args):
    world_size = args.world_size
    server_addr = (args.host, args.port)
    redundant = args.redundant
    auth_key = args.auth_key
    experts_set = set(range(args.expert_num))
    experts_per_rank = args.expert_num // world_size

    num_moe_layers = args.block_num
    load_report_buffer = {idx: {} for idx in range(num_moe_layers)}
    local_expert_buffer = {idx: {} for idx in range(num_moe_layers)}

    count = 0

    global upload_queues, instruction_queues
    # zmq
    for rank in range(world_size):
        upload_queues[rank] = queue.Queue()
        instruction_queues[rank] = queue.Queue()

    start_manager_server(server_addr, auth_key)

    logger.debug(f"[Scheduler] starting moniter")

    while True:
        all_queues_empty = True
        try:
            for rank in range(world_size):
                try:
                    report = upload_queues[rank].get_nowait() 
                    
                    layer_idx = report['moe_layer_idx']
                    load_data = report['load']
                    local_expert_list = report['local_expert_list']

                    if args.mode == "EX" and redundant > 0 and len(local_expert_list) != (experts_per_rank + redundant):
                        random_range = list(experts_set - set(local_expert_list))
                        redundant_expert = random.sample(random_range, redundant)
                        local_expert_list = local_expert_list + redundant_expert

                    load_report_buffer[layer_idx][rank] = load_data
                    local_expert_buffer[layer_idx][rank] = local_expert_list
                    transfer = UpdateTaskTransfer(instruction_queues, layer_idx)

                    all_queues_empty = False

                    if len(load_report_buffer[layer_idx]) == world_size:

                        response = load_report_buffer[layer_idx]
                        expert_dict = local_expert_buffer[layer_idx]
                        expert_dict = dict(sorted(expert_dict.items()))
                        load_report_buffer[layer_idx] = {} 
                        local_expert_buffer[layer_idx] = {}

                        logger.debug(f"[greedy] eplb greedy compute")
                        result = eplb_greedy(
                            response=response, algorithm_type=args.mode, 
                            device_to_expert=expert_dict, world_size=world_size, 
                            expert_num=args.expert_num, max_move=args.max_move, redundant=redundant)
                        (   
                            update, 
                            device_indices_list, 
                            local_expert_indices_list, 
                            local_expert_list, 
                            expert_trans_tensor
                        ) = result

                        if not update:
                            continue

                        transfer.update_emit_task(
                            device_indices_list, 
                            local_expert_indices_list, 
                            local_expert_list, 
                            expert_trans_tensor, 
                            world_size
                        )
                        count += 1
                        logger.info(f"[Scheduler] layer_{layer_idx} layout has computed.")
                except queue.Empty:
                    pass
                except Exception as e:
                    raise ModelExecError("[Scheduler] error : {e}") from e
        except (KeyboardInterrupt, SystemExit):
            logger.info("[Scheduler] exit sign!")
            break
        if all_queues_empty:
            time.sleep(0.1)
    
    logger.info(f"Already has update {count} times")
    logger.info("[Scheduler] Scheduler cycle end.")

if __name__ == '__main__':
    args = get_args()
    run_scheduler(args)