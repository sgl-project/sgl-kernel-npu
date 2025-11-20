from typing import Optional
from inspect import signature
import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl
import pytest

from sgl_kernel_npu.cumsum import chunk_local_cumsum_scalar_npu
from sgl_kernel_npu.utils.index import prepare_chunk_indices, prepare_chunk_offsets


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
        out_gpu = chunk_local_cumsum_scalar_gpu(
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
        out_gpu.cpu(),
        msg=f"GPU and NPU outputs differ! "
            f"shape={input_tensor.shape}, chunk_size={chunk_size}, "
            f"reverse={reverse}, scale={scale}, head_first={head_first}, "
            f"cu_seqlens={'provided' if cu_seqlens is not None else 'None'}"
    )


@pytest.mark.parametrize("input_info", [
    ((1, 8, 7168), torch.tensor([0, 3584, 7168], dtype=torch.long)),

    ])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("chunk_size", [64])
def test_chunk_local_cumsum_var_len(
    input_info, 
    dtype, 
    head_first, 
    reverse,
    scale=None,
    chunk_size=64
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

    _run_and_compare(
        input_tensor=x,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        output_dtype=dtype,
        atol=1e-3 if dtype == torch.float16 else 1e-5,
        rtol=1e-3 if dtype == torch.float16 else 1e-5,
    )


@pytest.mark.parametrize("shape", [
        (1, 8, 1024),
        (2, 8, 2048),
        (2, 16, 2048),
        (4, 16, 4096),
    ])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("chunk_size", [64])
def test_chunk_local_cumsum_fixed_len(
    shape, 
    dtype, 
    head_first, 
    reverse,
    scale=None,
    chunk_size=64
    ):
    device = "npu"
    B, H, T = shape
    if head_first:
        shape = (B, H, T)
    else:
        shape = (B, T, H)

    x = torch.randn(shape, dtype=dtype, device=device)

    _run_and_compare(
        input_tensor=x,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=dtype,
        atol=1e-3 if dtype == torch.float16 else 1e-5,
        rtol=1e-3 if dtype == torch.float16 else 1e-5,
    )


if __name__ == "__main__":
    torch_npu_flag = False
    if not torch.cuda.is_available():
        from torch_npu.contrib import transfer_to_npu
        torch_npu_flag = True


    def chunk_local_cumsum(
        g: torch.Tensor,
        chunk_size: int,
        reverse: bool = False,
        scale: float = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        head_first: bool = False,
        output_dtype: Optional[torch.dtype] = torch.float,
        **kwargs,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            assert (
                g.shape[0] == 1
            ), "Only batch size 1 is supported when cu_seqlens are provided"
        if len(g.shape) == 3:
            if torch_npu_flag:
                chunk_local_cumsum_scalar = chunk_local_cumsum_scalar_npu
            else:
                chunk_local_cumsum_scalar = chunk_local_cumsum_scalar_gpu
            return chunk_local_cumsum_scalar(
                g=g,
                chunk_size=chunk_size,
                reverse=reverse,
                scale=scale,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
                output_dtype=output_dtype,
            )
        elif len(g.shape) == 4:
            return chunk_local_cumsum_vector(
                g=g,
                chunk_size=chunk_size,
                reverse=reverse,
                scale=scale,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
                output_dtype=output_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported input shape {g.shape}, "
                f"which should be (B, T, H, D) if `head_first=False` "
                f"or (B, H, T, D) otherwise"
            )
    _CONDITIONS = ("seq7168", )
    for cond in _CONDITIONS:
        args, kwargs = torch.load(f"./chunk_local_cumsum@{cond}_input.pt", map_location="cuda")
        output = torch.load(f"./chunk_local_cumsum@{cond}_output.pt", map_location="cuda")

        print(signature(chunk_local_cumsum).bind(*args, **kwargs))

        result = chunk_local_cumsum(*args, **kwargs)
        # print(result - output, output.shape, result.shape)
        torch.testing.assert_close(result, output)
        if not torch_npu_flag:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=8, repeat=1, skip_first=0),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
            ) as prof:

                for i in range(20):
                    result = chunk_local_cumsum(*args, **kwargs)
                torch.cuda.synchronize()

            prof.export_chrome_trace(f"./chunk_local_cumsum.json")

        else:
            import torch_npu
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=8, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./chunk_local_cumsum"),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            ) as prof:

                for i in range(20):
                    result = chunk_local_cumsum(*args, **kwargs)
                torch.npu.synchronize()
