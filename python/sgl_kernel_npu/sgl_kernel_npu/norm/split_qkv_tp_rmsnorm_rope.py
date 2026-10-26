"""Compatibility path for TP split-QKV/RMSNorm/RoPE."""

from ..operators.norm.split_qkv_tp_rmsnorm_rope import split_qkv_tp_rmsnorm_rope


def __getattr__(name):
    from ..implementation.triton.norm import split_qkv_tp_rmsnorm_rope as implementation

    return getattr(implementation, name)
