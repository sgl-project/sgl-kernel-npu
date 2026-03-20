import numpy as np
import torch
from sgl_kernel_npu.norm.scale_shift import fused_scale_shift


def fused_scale_shift_golden(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
):
    return x * (1 + scale) + shift


def test_fused_scale():
    B, H, C = 3, 37440, 5120
    block_l, block_c = 128, 128
    dtype = torch.float32

    test_cases = [
        [
            (1, 1, C),
            (1, 1, C),
        ],
        [
            (1,),
            (1,),
        ],
        [
            (1, 1, C),
            (B, H, C),
        ],
    ]

    for shape in test_cases:
        print(f"Testing with scale/shift shape: {shape}")

        x = torch.randn(B, H, C, dtype=dtype, device="npu")
        scale = torch.randn(shape[0], dtype=dtype, device="npu")
        shift = torch.randn(shape[1], dtype=dtype, device="npu")

        res = fused_scale_shift_golden(x, scale, shift)
        ans = fused_scale_shift(x, scale, shift, block_l=block_l, block_c=block_c)

        np.testing.assert_allclose(
            res.cpu().numpy(),
            ans.cpu().numpy(),
            rtol=1e-3,
            err_msg=f"Failed for shape {shape}",
        )

        print(f"Passed: shape {shape}")


if __name__ == "__main__":
    test_fused_scale()
