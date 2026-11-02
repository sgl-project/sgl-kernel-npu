"""cann_ops_transformer adapter for compressor."""

import torch


def compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    # Both A3 and A5 bind the released/native compressor through the npu namespace.
    # Do not redirect this to the similarly named experimental custom op.
    return torch.ops.npu.compressor(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )
