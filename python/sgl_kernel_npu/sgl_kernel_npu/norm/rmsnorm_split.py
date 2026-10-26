"""Compatibility path for the migrated normalization operators."""

from ..operators.norm.fused_variance import fused_variance


def __getattr__(name):
    from ..implementation.triton.norm import rmsnorm_split as implementation

    return getattr(implementation, name)


__all__ = ["fused_variance", "fused_rsqrt_mul"]
