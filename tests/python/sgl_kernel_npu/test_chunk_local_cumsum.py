from inspect import signature
from typing import Optional

import torch
import torch_npu
import triton
import triton.language as tl
import pytest

from sgl_kernel_npu.fla.cumsum import chunk_local_cumsum_scalar_npu
from sgl_kernel_npu.utils.index import prepare_chunk_indices


#---------------------orig-----------------------------
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel_gpu(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_scalar_gpu(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel_gpu[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=8,
        num_stages=3,
    )
    return g


def chunk_local_cumsum_fix_len_torch(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:

    if head_first:
        B, H, T = g.shape
        g = g.transpose(1, 2)  # -> (B, T, H) for easier processing
    else:
        B, T, H = g.shape

    BT = chunk_size
    assert BT > 0 and (BT & (BT - 1)) == 0, "chunk_size must be a power of 2"

    # Output tensor
    output_dtype = output_dtype or g.dtype
    out = torch.empty_like(g, dtype=torch.float)
    g = g.to(torch.float)
    if cu_seqlens is not None:
        raise ValueError("This func does not support cu_seqlens")
    else:
        for b in range(B):
            seq = g[b]  # (T, H)
            for c in range(0, T, BT):
                e = min(c + BT, T)
                chunk = seq[c:e]  # (chunk_L, H)

                cum = torch.cumsum(chunk, dim=0)
                if reverse:
                    total = torch.sum(chunk, dim=0, keepdim=True)
                    cum = -cum + total + chunk
                if scale is not None:
                    cum = cum * scale

                out[b, c:e] = cum.to(output_dtype)

    if head_first:
        out = out.transpose(1, 2)  # back to (B, H, T)
    return out.to(output_dtype)


def _run_and_compare(
    input_tensor: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    head_first: bool,
    output_dtype: torch.dtype,
):
    with torch.no_grad():
        on_gpu = chunk_local_cumsum_scalar_gpu(
            g=input_tensor.clone(),
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )

    with torch.no_grad():
        out_npu = chunk_local_cumsum_scalar_npu(
            g=input_tensor.clone(),
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )

    torch.testing.assert_close(
        out_npu.cpu(),
        on_gpu.cpu(),
        atol = 1e-5,
        rtol = 1e-3,
        msg=f"GPU and NPU outputs differ! "
            f"{out_npu=}, {on_gpu=}"
            f"shape={input_tensor.shape}, chunk_size={chunk_size}, "
            f"reverse={reverse}, scale={scale}, head_first={head_first}, "
            f"cu_seqlens={'provided' if cu_seqlens is not None else 'None'}"
    )

def _run_and_compare_torch(
    input_tensor: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    head_first: bool,
    output_dtype: torch.dtype,
):
    with torch.no_grad():
        ref = chunk_local_cumsum_fix_len_torch(
            g=input_tensor.clone(),
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )

    with torch.no_grad():
        out_npu = chunk_local_cumsum_scalar_npu(
            g=input_tensor.clone(),
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )

    torch.testing.assert_close(
        out_npu.cpu(),
        ref.cpu(),
        atol = 1e-5,
        rtol = 1e-3  
    )


@pytest.mark.parametrize("input_info", [
    # Original case: two equal-length sequences, total length 7168, each segment 3584
    ((1, 8, 7168), torch.tensor([0, 3584, 7168], dtype=torch.long)),
    # Single sequence (simplest case)
    ((1, 8, 1024), torch.tensor([0, 1024], dtype=torch.long)),
    # Many small sequences (stress-test chunk partitioning)
    ((1, 8, 512), torch.tensor([0, 64, 128, 192, 256, 320, 384, 448, 512], dtype=torch.long)),
    # All sequence lengths are exact multiples of chunk_size (verify alignment)
    ((1, 8, 384), torch.tensor([0, 128, 256, 384], dtype=torch.long)),  # 128 = 2 * 64
    # Simulate realistic LLM scenarios: common sequence lengths like 2048, 4096
    ((1, 8, 6144), torch.tensor([0, 2048, 6144], dtype=torch.long)),  # 2048 + 4096
   
   
    # All sequences of length 1 (extreme boundary case)
    ((1, 8, 5), torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)),
    # Sequence length < chunk_size (64) → test small-sequence boundary
    ((1, 8, 100), torch.tensor([0, 30, 100], dtype=torch.long)),
    # Total length not divisible by chunk_size (7168 % 64 = 0; this one isn't)
    ((1, 8, 1000), torch.tensor([0, 400, 1000], dtype=torch.long)),  # 1000 % 64 = 40
    # Extreme case: one very long + one very short sequence
    ((1, 8, 4096 + 10), torch.tensor([0, 10, 4106], dtype=torch.long)),
        # Three sequences with unequal lengths
    ((1, 8, 2000), torch.tensor([0, 500, 1200, 2000], dtype=torch.long)),
    ])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("reverse", [False])
@pytest.mark.parametrize("chunk_size", [64])
def test_chunk_local_cumsum_var_len(
    input_info, 
    dtype, 
    head_first, 
    reverse,
    chunk_size,
    scale=None
    ):
    device = "npu"
    shape, cu_seqlens = input_info
    cu_seqlens = cu_seqlens.to(device)
    B, H, T = shape
    if head_first:
        shape = (B, H, T)
    else:
        shape = (B, T, H)

    x = torch.randn(shape, dtype=dtype, device=device)

    # _run_and_compare_torch(
    _run_and_compare(
        input_tensor=x,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        output_dtype=dtype,
    )


@pytest.mark.parametrize("shape", [
        (1, 8, 1024),
        (2, 8, 2048),
        (2, 16, 2048),
        (4, 16, 4096),
    ])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("chunk_size", [64])
def test_chunk_local_cumsum_fixed_len(
    shape, 
    dtype, 
    head_first, 
    reverse,
    chunk_size,
    scale=None
    ):
    device = "npu"
    B, H, T = shape
    if head_first:
        shape = (B, H, T)
    else:
        shape = (B, T, H)

    x = torch.randn(shape, dtype=dtype, device=device)

    _run_and_compare_torch(
        input_tensor=x,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=dtype,
    )