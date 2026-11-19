"""TorchNPU prefetch for an Ascend A3 weight-cache workload."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "data_move.prefetch"
_ARCH22 = {Generation.ARCH22}
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto", supported_generations=_ARCH22)
@register_impl(_OP_NAME, "torch_npu", supported_generations=_ARCH22)
def _torch_npu_impl(input, dependency, max_size, offset):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_prefetch

        _torch_npu_fn = npu_prefetch
    if offset == 0:
        return _torch_npu_fn(input, dependency, max_size)
    return _torch_npu_fn(input, dependency, max_size, offset=offset)


@register_operator(_OP_NAME)
def prefetch(input, dependency, max_size, offset=0, *, impl="auto"):
    """Request L2 prefetch for input with max_size and offset measured in bytes.

    The optional dependency tensor expresses execution order. Auto selects
    torch_npu on arch22. The operation returns None.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, dependency, max_size, offset)
