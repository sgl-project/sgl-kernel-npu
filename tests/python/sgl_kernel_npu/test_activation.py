import torch
from sgl_kernel_npu.activation.swiglu_silu_clamp_mul import (
    swiglu_silu_clamp_mul_native,
    swiglu_silu_clamp_mul_triton,
)


def test_swiglu_exact_match():
    device = "npu"
    dtype = torch.bfloat16

    test_shapes = [
        (1, 4096),
        (2, 4096),
        (4, 8192),
        (8, 8192),
    ]

    for shape in test_shapes:
        print(f"Testing shape: {shape}")

        torch.manual_seed(42)
        x = torch.randn(shape, device=device, dtype=dtype)

        out_native = swiglu_silu_clamp_mul_native(x)
        out_triton = swiglu_silu_clamp_mul_triton(x)

        assert (
            out_native.shape == out_triton.shape
        ), f"shape doesn't match native={out_native.shape} triton={out_triton.shape}"

        atol = 5e-2 if dtype == torch.bfloat16 else 1e-3
        assert torch.testing.assert_close(
            out_native, out_triton, atol=atol, rtol=1e-5
        ), f"value doesn't match shape={shape}"

        max_err = torch.max(torch.abs(out_native - out_triton))
        print(f"max_err {max_err:.6f}")

    print("\n Pass!")
