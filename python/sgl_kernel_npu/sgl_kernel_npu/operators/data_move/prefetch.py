"""TorchNPU prefetch for an Ascend A3 weight-cache workload."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "data_move.prefetch"
_ARCH22 = {Generation.ARCH22}


@register_impl(_OP_NAME, "auto", supported_generations=_ARCH22)
@register_impl(_OP_NAME, "runtime", supported_generations=_ARCH22)
def _torch_npu_impl(input, dependency, max_size, offset):
    if offset == 0:
        return torch_npu.npu_prefetch(input, dependency, max_size)
    return torch_npu.npu_prefetch(input, dependency, max_size, offset=offset)


@register_operator(_OP_NAME)
def prefetch(input, dependency, max_size, offset=0, *, impl="auto"):
    """Request L2 prefetch for input with max_size and offset measured in bytes.

    The optional dependency tensor expresses execution order. Auto selects
    runtime on arch22. The operation returns None.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, dependency, max_size, offset)
