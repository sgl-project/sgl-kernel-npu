"""PyTorch references for ordinary dtype conversion and dense ND layout."""

import torch


def dtype_cast(input, dtype, input_dtype=None):
    if not isinstance(dtype, torch.dtype) or input_dtype not in (None, input.dtype):
        raise NotImplementedError("torch_native only casts ordinary PyTorch dtypes")
    if dtype.is_complex or input.is_complex():
        raise NotImplementedError("torch_native does not model TorchNPU complex casts")
    return input.to(dtype=dtype)


def format_cast(input, acl_format, *, customize_dtype=None, input_dtype=None):
    if acl_format != 2 or customize_dtype is not None or input_dtype is not None:
        raise NotImplementedError("torch_native supports ordinary dense ND (format 2)")
    # A new dense tensor carries the logical values independently of the input
    # layout. PyTorch does not expose the other ACL physical layouts.
    return input.detach().cpu().contiguous().to(input.device, copy=True)
