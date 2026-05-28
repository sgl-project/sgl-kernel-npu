import logging
import os
import sys
from functools import reduce
from typing import List

import torch
from torch_memory_saver import torch_memory_saver
from torch_memory_saver.testing_utils import get_and_print_npu_memory


def run(hook_mode: str):
    assert os.environ["TMS_INIT_ENABLE"] == "1"
    assert os.environ["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"
    assert hook_mode == "preload"

    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    initial_tensor = torch.full((1_000_000,), 42, dtype=torch.uint8, device="npu")
    mem_initial = get_and_print_npu_memory("Init")

    model_weights = [
        torch.full((size,), 42, dtype=torch.uint8, device="npu")
        for size in [1024**3, 1024**2, 1024**1, 42]
    ]

    _execute_forward_pass_and_assert(model_weights)
    mem_after_forward_pass = get_and_print_npu_memory("After forward pass")
    assert mem_after_forward_pass > mem_initial + 5 * 1024**3

    torch_memory_saver.pause()
    mem_after_pause = get_and_print_npu_memory("After pause")
    assert mem_after_pause < mem_initial + 200 * 1024**2

    with torch_memory_saver.disable():
        mem_after_disable = get_and_print_npu_memory("After disable")
        assert mem_after_disable == mem_after_pause

        # Can still execute code in disabled region
        tensor_in_disabled_region = torch.full(
            (1024**3,), 53, dtype=torch.uint8, device="npu"
        )
        out = tensor_in_disabled_region.float().mean().item()
        assert out == 53, f"{out=}"

        mem_after_exec_in_disable = get_and_print_npu_memory("After exec in disable")
        assert mem_after_exec_in_disable > mem_after_disable + 4 * 1024**3

        del tensor_in_disabled_region

    """
    Currently, the behavior in this scenario is inconsistent with the GPU version of                                        torch_memory_saver. The NPU memory allocated within the disable() context is not                                        released immediately; it will only be freed when torch.npu.empty_cache() is invoked.
    Additionally, since empty_cache() takes effect for all memory pools, it is not                                          recommended to invoke it in the paused state, as this may corrupt the address
    spaces that have already been paused.

    Expected (GPU): mem_after_exit_disable <= mem_after_pause + 10 * 1024 ** 2
    NPU Current: mem_after_exit_disable == mem_after_exec_in_disable
    """
    mem_after_exit_disable = get_and_print_npu_memory("After exiting disable")
    assert mem_after_exit_disable == mem_after_exec_in_disable

    torch_memory_saver.resume()
    mem_after_resume = get_and_print_npu_memory("After resume")
    delta_expect = mem_after_forward_pass - mem_after_pause
    delta_actual = mem_after_resume - mem_after_exit_disable
    assert delta_expect - 50 * 1024**2 < delta_actual < delta_expect + 50 * 1024**2

    _execute_forward_pass_and_assert(model_weights)
    mem_after_second_forward_pass = get_and_print_npu_memory(
        "After second forward pass"
    )
    assert (
        mem_after_resume - 1024**2
        < mem_after_second_forward_pass
        < mem_after_resume + 1024**2
    )

    del initial_tensor


def _execute_forward_pass_and_assert(weights: List[torch.Tensor]):
    # simulate large activations during forward pass
    ones = torch.ones((1024**3,), dtype=torch.float32, device="npu")
    sum_avg_weights = reduce(
        lambda a, b: (a + b) / 2, [w.float().mean() for w in weights]
    )
    outs = ones * sum_avg_weights
    out = outs.mean().item()
    assert out == 42, f"{out=}"


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
