"""Compatibility exports for the NPU small-page allocator."""

from sgl_kernel_npu.implementation.triton.mem_cache.npu_allocator import (
    _page_offset as _page_offset,
)
from sgl_kernel_npu.implementation.triton.mem_cache.npu_allocator import (
    alloc_extend_kernel as alloc_extend_kernel,
)
