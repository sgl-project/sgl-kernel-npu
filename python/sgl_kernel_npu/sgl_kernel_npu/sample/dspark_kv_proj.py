"""DSPark commit KV projection with a tiled Triton-Ascend matmul."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _dequant_kernel(
    weight,
    scale,
    out,
    n,
    width,
    scale_stride,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        x = tile * BLOCK + tl.arange(0, BLOCK)
        if HAS_SCALE:
            # Decode E4M3 bytes: A2/A3 do not support native FP8 conversion.
            bits = tl.load(weight + x, x < n, other=0).to(tl.int32)
            exponent = (bits >> 3) & 15
            mantissa = bits & 7
            normal = ((exponent + 120) << 23) | (mantissa << 20)
            value = tl.where(
                exponent == 0, mantissa * (2.0**-9), normal.to(tl.float32, bitcast=True)
            )
            value = tl.where((bits & 128) != 0, -value, value)
            value = tl.where((bits & 127) == 127, float("nan"), value)
            sf = tl.load(
                scale + x // width // 128 * scale_stride + x % width // 128,
                x < n,
                other=0,
            ).to(tl.float32)
            value *= sf
        else:
            value = tl.load(weight + x, x < n, other=0).to(tl.float32)
        tl.store(out + x, value, x < n)


def _dequant_linear_weight(linear, dtype=torch.bfloat16):
    weight = linear.weight.contiguous()
    has_scale = weight.dtype == torch.float8_e4m3fn
    if has_scale:
        scale = linear.weight_scale_inv.contiguous()
        assert scale.dtype in (torch.float32, torch.bfloat16, torch.float16)
        assert tuple(scale.shape) == (
            triton.cdiv(weight.shape[0], 128),
            triton.cdiv(weight.shape[1], 128),
        )
    else:
        assert weight.dtype in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ), weight.dtype
        scale = None
    if not has_scale and weight.dtype == dtype:
        return weight
    out = torch.empty_like(weight, dtype=dtype)
    n = weight.numel()
    _dequant_kernel[max(1, min(get_device_properties()[1], triton.cdiv(n, 1024))),](
        weight.view(torch.uint8) if has_scale else weight,
        scale,
        out,
        n,
        weight.shape[1],
        scale.stride(0) if has_scale else 0,
        HAS_SCALE=has_scale,
        BLOCK=1024,
        num_warps=1,
    )
    return out


@triton.jit
def _project_kernel(
    x,
    w,
    out,
    rows,
    width,
    inner,
    xs0,
    xs1,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ncols = tl.cdiv(width, BN)
    for tile in range(tl.program_id(0), tl.cdiv(rows, BM) * ncols, tl.num_programs(0)):
        m = tile // ncols * BM + tl.arange(0, BM)
        n = tile % ncols * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(0, inner, BK):
            kk = start + k
            a = tl.load(
                x + m[:, None] * xs0 + kk[None, :] * xs1,
                (m[:, None] < rows) & (kk[None, :] < inner),
                other=0,
            )
            b = tl.load(
                w + n[None, :] * inner + kk[:, None],
                (n[None, :] < width) & (kk[:, None] < inner),
                other=0,
            )
            acc = tl.dot(a, b, acc, input_precision="ieee")
        tl.store(
            out + m[:, None] * width + n[None, :],
            acc,
            (m[:, None] < rows) & (n[None, :] < width),
        )


def commit_kv_proj(*, main_x, wkv_linears, allow_strided_output=False):
    assert main_x.dtype in (torch.bfloat16, torch.float16, torch.float32), main_x.dtype
    rows, inner = main_x.shape
    outputs = []
    for linear in wkv_linears:
        weight = _dequant_linear_weight(linear, dtype=main_x.dtype)
        assert weight.shape[1] == inner
        width = weight.shape[0]
        out = torch.empty((rows, width), dtype=main_x.dtype, device=main_x.device)
        bm, bn, bk = 16, 64, 64
        tiles = triton.cdiv(rows, bm) * triton.cdiv(width, bn)
        _project_kernel[max(1, min(get_device_properties()[0], tiles)),](
            main_x,
            weight,
            out,
            rows,
            width,
            inner,
            main_x.stride(0),
            main_x.stride(1),
            BM=bm,
            BN=bn,
            BK=bk,
            num_warps=1,
        )
        outputs.append(out)
    return outputs


class CommitKvProj:
    execute = staticmethod(commit_kv_proj)
    triton = staticmethod(commit_kv_proj)
