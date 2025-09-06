import logging
import sys
import time

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_npu_memory


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    normal_tensor = torch.full((1_000_000,), 100, dtype=torch.uint8, device="npu")
    with torch_memory_saver.region(enable_cpu_backup=False):
        pauseable_tensor = torch.full(
            (1_000_000_000,), 100, dtype=torch.uint8, device="npu"
        )
    torch.npu.synchronize()
    original_address = pauseable_tensor.data_ptr()
    print(f"Pauseable tensor virtual address: 0x{original_address:x}")
    print(f"{normal_tensor=} {pauseable_tensor=}")

    mem_before_pause = get_and_print_npu_memory("Before pause")

    torch_memory_saver.pause()

    mem_after_pause = get_and_print_npu_memory("After pause")

    assert mem_before_pause - mem_after_pause > 0.9 * 1024**3

    print("sleep...")
    time.sleep(1)

    torch_memory_saver.resume()

    mem_after_resume = get_and_print_npu_memory("After resume")

    assert mem_after_resume - mem_after_pause > 0.9 * 1024**3

    new_address = pauseable_tensor.data_ptr()
    print(f"Pauseable tensor virtual address: 0x{new_address:x}")

    assert original_address == new_address, "Tensor virtual address should be the same"

    print("sleep...")
    time.sleep(1)
    torch.npu.synchronize()
    print(f"{normal_tensor=} {pauseable_tensor=}")

    get_and_print_npu_memory("Before empty cache")
    torch.npu.empty_cache()
    get_and_print_npu_memory("After empty cache")

    del normal_tensor, pauseable_tensor

    get_and_print_npu_memory("Before empty cache (tensor deleted)")
    torch.npu.empty_cache()
    get_and_print_npu_memory("After empty cache (tensor deleted)")
    torch.npu.synchronize()


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
