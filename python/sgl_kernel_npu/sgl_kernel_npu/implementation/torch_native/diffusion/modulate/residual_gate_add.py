"""Eager reference for the fused residual gate add."""


def residual_gate_add(residual, update, gate):
    """Compute ``residual + update * gate`` as two eager BF16 operations."""
    return residual + update * gate
