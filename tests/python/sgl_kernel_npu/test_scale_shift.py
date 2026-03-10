import numpy as np
import torch
from sgl_kernel_npu.norm.scale_shift import fuse_scale_shift

def fuse_scale_shift_golden(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
):
    return x * (1 + scale) + shift

def test_fuse_scale():
    B, H, C = 3, 37440, 5120
    block_l, block_c = 128, 128
    dtype = torch.float32

    test_cases = [
        (1, 1, C),
        (1,),
    ]

    for shape in test_cases:
        print(f"Testing with scale/shift shape: {shape}")

        x = torch.randn(B, H, C, dtype=dtype, device="npu")
        scale = torch.randn(shape, dtype=dtype, device="npu")
        shift = torch.randn(shape, dtype=dtype, device="npu")

        res = fuse_scale_shift_golden(x, scale, shift)
        ans = fuse_scale_shift(x, scale, shift, block_l, block_c)

        np.testing.assert_allclose(
            res.cpu().numpy(),
            ans.cpu().numpy(),
            rtol=1e-1,
            err_msg=f"Failed for shape {shape}"
        )

        print(f"Passed: shape {shape}")

if __name__ == "__main__":
    test_fuse_scale()
