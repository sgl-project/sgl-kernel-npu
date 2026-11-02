"""cann_ops_transformer adapter for kv_compress_epilog."""

import torch

from .._ops import get_op


def kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    get_op("kv_compress_epilog")(cache.view(torch.uint8), x, slot_mapping, **kwargs)
    return cache
