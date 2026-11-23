"""CPU PyTorch references for the migrated TorchNPU quantized GEMMs."""

import torch

from ..attention.mlaprolog_v4 import _cpu, _decode_e8m0


def quant_matmul(
    x1,
    x2,
    scale,
    *,
    pertoken_scale=None,
    bias=None,
    output_dtype=None,
):
    if (
        x1.dtype != torch.int8
        or x2.dtype != torch.int8
        or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise NotImplementedError(
            "torch_native requires INT8 inputs and floating scale"
        )
    product = _cpu(x1).to(torch.int32) @ _cpu(x2).to(torch.int32)
    if bias is not None and bias.dtype == torch.int32:
        product = product + _cpu(bias)
    result = product.float() * _cpu(scale).float()
    if pertoken_scale is not None:
        token_scales = _cpu(pertoken_scale).reshape(*product.shape[:-1], 1).float()
        result = result * token_scales
    if bias is not None and bias.dtype != torch.int32:
        result = result + _cpu(bias).float()
    dtype = torch.int8 if output_dtype is None else output_dtype
    if dtype == torch.int8:
        result = result.round().clamp(-128, 127)
    elif dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "torch_native supports INT8, FP16, BF16 and FP32 output"
        )
    return result.to(dtype).to(x1.device)


def weight_quant_batchmatmul(
    x,
    weight,
    antiquant_scale,
    antiquant_offset=None,
    quant_scale=None,
    quant_offset=None,
    bias=None,
    antiquant_group_size=0,
    inner_precise=0,
    weight_dtype=None,
):
    if (
        weight.dtype != torch.int8
        or antiquant_group_size < 0
        or quant_scale is not None
        or quant_offset is not None
        or weight_dtype is not None
        or inner_precise not in (0, 1)
    ):
        raise NotImplementedError(
            "torch_native requires unquantized output and INT8 weights"
        )
    weights = _cpu(weight).float()
    scales = _cpu(antiquant_scale).float()
    offsets = _cpu(antiquant_offset).float() if antiquant_offset is not None else None
    if antiquant_group_size:
        width = weights.size(-2)
        scales = scales.repeat_interleave(antiquant_group_size, dim=-2)[..., :width, :]
        if offsets is not None:
            offsets = offsets.repeat_interleave(antiquant_group_size, dim=-2)[
                ..., :width, :
            ]
    if offsets is not None:
        weights = weights + offsets
    weights = weights * scales
    result = _cpu(x).float() @ weights
    if bias is not None:
        result = result + _cpu(bias).float()
    return result.to(x.dtype).to(x.device)


def _fp8_float(tensor):
    return tensor.detach().view(torch.uint8).cpu().view(tensor.dtype).float()


def transpose_quant_batchmatmul(
    x1,
    x2,
    dtype,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=1,
):
    if (
        x1.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or x2.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or x1_scale is None
        or x2_scale is None
        or tuple(group_sizes or ()) != (0, 0, 32)
        or (tuple(perm_x1 or (0, 1, 2)), tuple(perm_x2 or (0, 1, 2)))
        not in (((1, 0, 2), (0, 1, 2)), ((0, 1, 2), (0, 1, 2)))
    ):
        raise NotImplementedError("torch_native supports grouped MXFP8 WoA layouts")

    def decode(scale):
        bits = scale.detach().view(torch.uint8).cpu()
        return _decode_e8m0(bits)

    x1_groups = decode(x1_scale).flatten(-2).repeat_interleave(32, dim=-1)
    x2_groups = (
        decode(x2_scale).transpose(-1, -2).flatten(1, 2).repeat_interleave(32, dim=1)
    )
    lhs = _fp8_float(x1) * x1_groups[..., : x1.size(-1)]
    rhs = _fp8_float(x2) * x2_groups[:, : x2.size(1)]
    result = lhs.permute(perm_x1 or (0, 1, 2)) @ rhs
    if bias is not None:
        result = result + _cpu(bias).float()
    return result.permute(perm_y or (0, 1, 2)).contiguous().to(dtype).to(x1.device)


def _dequantize_fp4(data, level0, level1):
    # `_cpu` unpacks the right operand's NZ storage before the logical decode.
    packed = _cpu(data).view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    magnitudes = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device="cpu"
    )
    values = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    l0 = _cpu(level0).repeat_interleave(512, dim=-1)
    scale_bytes = level1.detach().view(torch.uint8).cpu()
    l1 = _decode_e8m0(scale_bytes).flatten(-2).repeat_interleave(32, dim=-1)
    return values * l1[..., : values.size(-1)] * l0[..., : values.size(-1)]


def dual_level_quant_matmul(
    x1,
    x2,
    x1_level0_scale,
    x2_level0_scale,
    x1_level1_scale,
    x2_level1_scale,
    *,
    bias=None,
    output_dtype=torch.bfloat16,
):
    if (
        x1.ndim != 2
        or x2.ndim != 2
        or any(
            value.dtype not in (torch.uint8, torch.int8, torch.float4_e2m1fn_x2)
            for value in (x1, x2)
        )
    ):
        raise NotImplementedError("torch_native requires packed FP4 matrices")
    lhs = _dequantize_fp4(x1, x1_level0_scale, x1_level1_scale)
    rhs = _dequantize_fp4(x2, x2_level0_scale.transpose(0, 1), x2_level1_scale)
    result = lhs @ rhs.T
    if bias is not None:
        result = result + _cpu(bias).float()
    return result.to(output_dtype).to(x1.device)
