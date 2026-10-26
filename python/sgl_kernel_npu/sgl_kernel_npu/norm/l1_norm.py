"""Compatibility path for the migrated normalization operators."""

from ..operators.norm.l1_norm import l1_norm


def __getattr__(name):
    from ..implementation.triton.norm import l1_norm as implementation

    return getattr(implementation, name)


__all__ = ["l1_norm"]
