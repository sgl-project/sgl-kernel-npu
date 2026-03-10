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
    dtype = torch.bfloat16

    test_cases = [
        (1, 1, C),
        (1,),
    ]

    for shape in test_cases:
        x = torch.randn(B, H, C, dtype=torch.float32, device="npu")
        scale = torch.randn(shape, dtype=dtype, device="npu")
        shift = torch.randn(shape, dtype=dtype, device="npu")

        res = fuse_scale_shift_golden(x, scale, shift)
        ans = fuse_scale_shift(x, scale, shift, block_l, block_c)

        assert (
            np.testing.assert_allclose(res, ans, rtol=1e-1) is None
        )

if __name__ == "__main__":
    test_fuse_scale()
