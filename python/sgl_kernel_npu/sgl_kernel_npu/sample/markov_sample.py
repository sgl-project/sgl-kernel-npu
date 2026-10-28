"""Compatibility imports for the sampling operator."""

from sgl_kernel_npu.implementation.triton.sample.markov_sample import *  # noqa: F401,F403
from sgl_kernel_npu.implementation.triton.sample.markov_sample import (  # noqa: F401
    _negative_log1p_negative,
)
