import numpy as np
import torch
from sgl_kernel_npu.norm.rmsnorm_split import fused_rsqrt_mul, fused_variance


def rsqrt_mul_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance: torch.Tensor,
    eps: float,
):
    return x * torch.rsqrt(variance + eps) * weight
    

def variance_golden(x: torch.Tensor):
    return x.pow(2).mean(dim=-1, keepdim=True)


def test_fused_rsqrt_mul():
    B, L, C = 1, 8190, 2560
    dtype = torch.float32

    x = torch.randn(B, L, C, dtype=dtype, device="npu")
    weight = torch.randn(C, device="npu")
    variance = torch.randn(1, L, 1, device="npu").abs() + 0.1
    eps = 1e-6

    res = rsqrt_mul_golden(x, weight, variance, eps)
    ans = fused_rsqrt_mul(x, variance.view(-1), weight, eps)

    np.testing.assert_allclose(
        res.cpu().numpy(),
        ans.cpu().numpy(),
        rtol=1e-3,
        err_msg="Fused_rsqrt_mul failed!"
    )

    print("Fused_rsqrt_mul Passed!")


def test_fused_variance():
    B, L, C = 1, 1024, 512
    dtype = torch.float32

    x = torch.randn(B, L, C, dtype=dtype, device="npu")
    res = variance_golden(x)
    ans = fused_variance(x)

    np.testing.assert_allclose(
        res.cpu().numpy(),
        ans.cpu().numpy(),
        rtol=1e-3,
        err_msg="Fused_variance failed!"
    )

    print("Fused_variance Passed!")


if __name__ == "__main__":
    test_fused_rsqrt_mul()
    test_fused_variance()