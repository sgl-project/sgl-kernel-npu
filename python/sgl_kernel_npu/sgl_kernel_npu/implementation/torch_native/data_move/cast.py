"""PyTorch references for dtype conversion, FP4 weight packing and dense ND layout."""

import torch


def dtype_cast(input, dtype, input_dtype=None):
    if isinstance(dtype, torch.dtype):
        packed_fp4 = dtype is getattr(torch, "float4_e2m1fn_x2", None)
    else:
        import torch_npu

        packed_fp4 = dtype is not None and dtype == getattr(
            torch_npu, "float4_e2m1fn_x2", None
        )
    if packed_fp4:
        from ..quant.dynamic import _pack_e2m1

        if input.dtype != torch.float8_e4m3fn or input_dtype not in (None, input.dtype):
            raise NotImplementedError("torch_native packs E4M3 weights to E2M1")
        if input.ndim == 0 or input.size(-1) % 2:
            raise ValueError("Packed FP4 weights require an even final dimension")
        values = input.contiguous().view(torch.uint8).cpu().view(input.dtype)
        return _pack_e2m1(values.float()).to(input.device)
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
