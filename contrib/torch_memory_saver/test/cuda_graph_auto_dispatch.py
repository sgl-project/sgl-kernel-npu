"""Capture a pauseable graph with ``auto_dispatch_capture`` and replay it.

``torch.npu.graph`` accepts ``auto_dispatch_capture`` so the NPU dispatch mode
can capture ops that plain capture misses; sglang's NPU graph runner always
passes it, so the wrapper must forward it instead of rejecting the keyword.

Run on an NPU host: ``python test/cuda_graph_auto_dispatch.py``
"""

import logging
import sys
import time

import torch

from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_npu_memory

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

kv_shape = (5, 100_000_000)
intermediate_shape = (1_000_000_000,)


class KVCache:
    def __init__(self):
        with torch_memory_saver.region(tag="kv_cache"):
            self.kv_buffer = torch.full(kv_shape, 1, dtype=torch.float32, device="npu")

    def execute(self, arg: torch.Tensor) -> torch.Tensor:
        value = (arg + self.kv_buffer.mean(dim=1)).mean()
        return (torch.ones(intermediate_shape, device="npu") * value).mean()


def capture(fn) -> torch.npu.NPUGraph:
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        fn()
    torch.npu.current_stream().wait_stream(stream)

    graph = torch.npu.NPUGraph()
    with torch_memory_saver.cuda_graph(
        graph, tag="graph", auto_dispatch_capture=True
    ):
        fn()
    return graph


def main() -> None:
    torch_memory_saver.hook_mode = "preload"

    cache = KVCache()
    static_input = torch.zeros((5,), dtype=torch.float32, device="npu")
    result = {}

    def fn():
        result["out"] = cache.execute(static_input)

    graph = capture(fn)

    static_input[...] = 100
    graph.replay()
    assert result["out"] == 101, result["out"]
    print("replay with auto_dispatch_capture=True:", float(result["out"]))

    torch.npu.empty_cache()
    time.sleep(1)
    before = get_and_print_npu_memory("before pause")
    torch_memory_saver.pause("kv_cache")
    time.sleep(1)
    after = get_and_print_npu_memory("after pause kv_cache")
    assert before - after > 400_000_000, (before, after)
    torch_memory_saver.resume("kv_cache")

    print("PASS auto_dispatch_capture forwarded and graph is pauseable")


if __name__ == "__main__":
    main()
