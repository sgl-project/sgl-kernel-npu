"""cann_ops_transformer adapter for inplace_partial_rotary_mul."""

from .._ops import get_op


def inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    get_op("inplace_partial_rotary_mul")(x, cos, sin, **kwargs)
    return x
