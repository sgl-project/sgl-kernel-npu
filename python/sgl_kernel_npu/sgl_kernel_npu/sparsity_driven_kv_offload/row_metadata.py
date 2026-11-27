"""Compatibility entry point for sparse KV row metadata."""

from sgl_kernel_npu.implementation.triton.mem_cache.copy_rows_metadata import (
    copy_rows_metadata,
)

__all__ = ["copy_rows_metadata"]
