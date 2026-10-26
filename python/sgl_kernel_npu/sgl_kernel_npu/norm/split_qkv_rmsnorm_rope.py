"""Compatibility path for the migrated normalization operators."""

from ..operators.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
from ..operators.norm.split_qkvgate_gemma_rmsnorm_rope import (
    split_qkvgate_gemma_rmsnorm_rope,
)


def __getattr__(name):
    from ..implementation.triton.norm import split_qkv_rmsnorm_rope as implementation

    return getattr(implementation, name)


__all__ = ["split_qkv_rmsnorm_rope", "split_qkvgate_gemma_rmsnorm_rope"]
