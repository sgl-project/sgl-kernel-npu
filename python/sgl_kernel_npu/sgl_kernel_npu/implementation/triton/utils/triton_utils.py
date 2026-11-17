from functools import cache
from typing import Tuple

import torch
import torch_npu


@cache
@torch.compiler.assume_constant_result
def get_device_properties() -> Tuple[int, int]:
    """Return the cached Cube and Vector core counts for this process."""
    device = torch_npu.npu.current_device()
    properties = torch_npu.npu.get_device_properties(device)

    num_aicore = properties.cube_core_num
    num_vectorcore = properties.vector_core_num

    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


def _floor_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x).bit_length() - 1)
