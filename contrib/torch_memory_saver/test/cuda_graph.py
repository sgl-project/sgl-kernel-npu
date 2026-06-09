import logging
import os
import sys
import time
from typing import Callable

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_npu_memory

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

dummy_tensor_size = (
    5,
    100_000_000,
)
cuda_graph_intermediate_tensor_size = (1_000_000_000,)


def _ptr(x):
    assert isinstance(x, torch.Tensor)
    return hex(x.data_ptr())


class KVCache:
    def __init__(self):
        self.create_buffers(1)

    def create_buffers(self, value):
        with torch_memory_saver.region(tag="kv_cache"):
            # or model weights, etc
            self.kv_buffer = torch.full(
                dummy_tensor_size, value, dtype=torch.float32, device="npu"
            )
        print(f"create_buffers {_ptr(self.kv_buffer)=}")

    def clear_buffers(self):
        del self.kv_buffer

    def execute(self, arg: torch.Tensor) -> torch.Tensor:
        # print(f'KVCache.execute {arg=} {self.kv_buffer=}')
        ans_value = (arg + self.kv_buffer.mean(dim=1)).mean()
        big_intermediate_tensor = (
            torch.ones(cuda_graph_intermediate_tensor_size, device="npu") * ans_value
        ).mean()
        return big_intermediate_tensor


def create_cuda_graph(fn: Callable, hook_mode):
    # warmup
    s = torch.npu.Stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        print("with torch.npu.stream(s) execute fn")
        fn()
    torch.npu.current_stream().wait_stream(s)

    # capture
    g = torch.npu.NPUGraph()
    ctx = (
        torch_memory_saver.cuda_graph(g, tag="graph")
        if hook_mode == "preload"
        else torch.npu.graph(g)
    )
    with ctx:
        print("with torch.npu.graph(g) execute fn")
        fn()

    return g


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    cache = KVCache()
    static_input = torch.zeros((5,), dtype=torch.float32, device="npu")
    static_output = torch.zeros((5,), dtype=torch.float32, device="npu")
    print(f"{_ptr(static_input)=} {_ptr(static_output)=}")

    def fn():
        nonlocal static_output
        static_output = cache.execute(static_input)

    g = create_cuda_graph(fn, hook_mode=hook_mode)

    print("replay #1")
    static_input[...] = 100
    g.replay()
    print(f"{static_output=}")
    assert static_output == 101, f"{static_output=}"

    print("torch.npu.empty_cache()")
    torch.npu.empty_cache()

    print("sleep...")
    time.sleep(1)

    mem_before_pause = get_and_print_npu_memory("Before pause")

    print('call memory_saver.pause("kv_cache")')
    torch_memory_saver.pause("kv_cache")
    print("sleep...")
    time.sleep(1)

    mem_after_pause_kv_cache = get_and_print_npu_memory("After pause kv_cache")
    assert mem_before_pause - mem_after_pause_kv_cache > 400_000_000

    if hook_mode == "preload":
        print('call memory_saver.pause("graph")')
        torch_memory_saver.pause("graph")
        print("sleep...")
        time.sleep(1)

        mem_after_pause_graph = get_and_print_npu_memory("After pause graph")
        assert mem_after_pause_kv_cache - mem_after_pause_graph > 3_000_000_000

    print("when kv cache is released, we can allocate *other* big tensors")
    other_big_tensor = torch.zeros((2500_000_000,), dtype=torch.uint8, device="npu")
    print("sleep...")
    time.sleep(1)
    print(f"{other_big_tensor=}")
    del other_big_tensor
    torch.npu.empty_cache()
    print("sleep...")
    time.sleep(1)

    if hook_mode == "preload":
        print('call memory_saver.resume("graph")')
        torch_memory_saver.resume("graph")
    print('call memory_saver.resume("kv_cache")')
    torch_memory_saver.resume("kv_cache")

    dummy = torch.zeros((3,), device="npu")
    print(f"{_ptr(dummy)=}")

    cache.kv_buffer[...] = 2

    print("replay #2")
    static_input[...] = 200
    g.replay()
    print(f"{static_output=}")
    assert static_output == 202, f"{static_output=}"

    print("sleep...")
    time.sleep(1)

    print(f"{dummy=}")


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
