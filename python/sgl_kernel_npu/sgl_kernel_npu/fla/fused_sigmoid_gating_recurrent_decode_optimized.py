"""Compatibility import for the migrated fused decode update."""

from sgl_kernel_npu.operators.attention import (
    fused_gdn_decode as fused_sigmoid_gating_delta_rule_update_decode_npu,
)
