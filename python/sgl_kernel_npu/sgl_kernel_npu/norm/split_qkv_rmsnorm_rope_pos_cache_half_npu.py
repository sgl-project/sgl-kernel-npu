"""Compatibility path for the migrated normalization operators."""

from ..operators.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
    split_qkv_rmsnorm_rope_pos_cache_half_npu,
)


def __getattr__(name):
    from ..implementation.triton.norm import (
        split_qkv_rmsnorm_rope_pos_cache_half_npu as implementation,
    )

    return getattr(implementation, name)


__all__ = ["split_qkv_rmsnorm_rope_pos_cache_half_npu"]
