"""cann_ops_transformer adapter for mhc_post."""

from .._ops import get_op


def mhc_post(x, residual, post, comb):
    return get_op("mhc_post")(residual, comb, x, post)
