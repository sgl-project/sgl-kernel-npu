from enum import IntEnum
from functools import lru_cache
from typing import Tuple

import torch
import torch_npu


class DeviceCapability(IntEnum):
    """
    Refer to `SocVersion` in https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NpuVariables.h

    Please be aware that this is an internal interface which is subjected to change without prior notice.
    """

    # Ascend 910B1,910B2,910B2C,910B3,910B4,910B4_1
    A2 = 220

    # Ascend 910_9391,910_9392,910_9381,910_9382,910_9372,910_9362,
    A3 = 250


@lru_cache(maxsize=1)
def get_device_name() -> str:
    device = torch.npu.current_device()
    return torch.npu.get_device_name(device)


@lru_cache(maxsize=1)
def get_device_properties() -> Tuple[int, int]:
    device = torch.npu.current_device()
    device_properties = torch.npu.get_device_properties(device)

    cube_core_num = device_properties.cube_core_num
    vector_core_num = device_properties.vector_core_num

    return cube_core_num, vector_core_num


@lru_cache(maxsize=1)
def get_device_capability() -> DeviceCapability:
    soc = torch_npu._C._npu_get_soc_version()
    soc = soc // 10 * 10

    assert soc in iter(DeviceCapability), f"Unsupported soc: {get_device_name()}"

    return DeviceCapability(soc)
