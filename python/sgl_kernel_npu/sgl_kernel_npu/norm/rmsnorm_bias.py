"""Compatibility path for the unified RMSNorm+bias operator."""

from ..operators.norm.rmsnorm_bias import rmsnorm_bias


def __getattr__(name):
    from ..implementation.triton.norm import rmsnorm_bias as implementation

    return getattr(implementation, name)
