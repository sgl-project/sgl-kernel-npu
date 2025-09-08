import logging
import sys
import time

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_npu_memory


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    checker = _MemoryChecker()

    torch.npu.set_device(1)

    with torch_memory_saver.region():
        dev0_a = torch.full((100_000_000,), 10, dtype=torch.uint8, device="npu:0")
    torch.npu.synchronize()
    checker.check_and_update("alloc dev0_a", min_delta=(80_000_000, 0))

    with torch_memory_saver.region():
        dev1_a = torch.full((100_000_000,), 10, dtype=torch.uint8, device="npu")
    torch.npu.synchronize()
    checker.check_and_update("alloc dev1_a", min_delta=(0, 80_000_000))

    with torch_memory_saver.region():
        dev1_b = torch.full((100_000_000,), 10, dtype=torch.uint8, device="npu:1")
    torch.npu.synchronize()
    checker.check_and_update("alloc dev1_b", min_delta=(0, 80_000_000))

    torch_memory_saver.pause()
    torch_memory_saver.resume()

    checker.check_and_update("End", min_delta=(0, 0))


class _MemoryChecker:
    def __init__(self):
        self._prev = self._get("Initial")

    def _get(self, message: str):
        return (
            get_and_print_npu_memory(message, npu_id=0),
            get_and_print_npu_memory(message, npu_id=1),
        )

    def check_and_update(self, message: str, min_delta):
        curr = self._get(message)
        assert all(
            (curr_i - prev_i) >= min_delta_i
            for curr_i, prev_i, min_delta_i in zip(
                curr, self._prev, min_delta, strict=True
            )
        )

        self._prev = curr


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
