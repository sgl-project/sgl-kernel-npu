from functools import cache
from typing import Tuple

import torch
import torch_npu
import triton
import triton.language as tl


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


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845
