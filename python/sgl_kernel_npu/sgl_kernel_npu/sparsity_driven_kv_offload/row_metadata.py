"""Compatibility entry point for sparse KV row metadata."""

from sgl_kernel_npu.implementation.triton.mem_cache.prefill_plan import prefill_plan

__all__ = ["prefill_plan"]
