"""Compatibility import for the migrated chunked state update."""

from sgl_kernel_npu.operators.attention import (
    chunk_delta_h as chunk_gated_delta_rule_fwd_h_npu,
)
