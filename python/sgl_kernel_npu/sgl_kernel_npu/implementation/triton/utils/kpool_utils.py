"""Shared KPool constants and Vector Core launch sizing."""

import torch
import triton

INDEX_HEAD_DIM = 128

KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _default_triton_num_programs(device: torch.device, n_pools: int) -> int:
    """Choose a Vector Core-sized Triton grid for ``n_pools`` work items.

    Args:
        device: NPU device that owns the input tensors.
        n_pools: Number of pooled keys to assemble.

    Returns:
        The number of Triton programs to launch. Each program processes one or
        more pools with a grid-stride loop when ``n_pools`` exceeds this value.
    """

    # Triton permits up to 65535 logical programs. Use that limit if the driver
    # does not expose the physical Vector Core count.
    fallback = min(n_pools, 65535)
    try:
        device_index = device.index
        if device_index is None and hasattr(torch, "npu"):
            device_index = torch.npu.current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(
            device_index
        )
        vector_core_count = int(properties["num_vectorcore"])
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return fallback
    if vector_core_count <= 0:
        return fallback
    return min(n_pools, vector_core_count)
