import numpy as np
import torch
import torch.nn.functional as F
from sgl_kernel_npu.norm.rmsnorm_without_weight import fused_rmsnorm_without_weight


def fused_rmsnorm_without_weight_golden(
    x: torch.Tensor,
    eps: float,
):
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps)


def test_fused_scale():
    B, H, C = 1, 130, 2048
    dtype = torch.float32

    x = torch.randn(B, H, C, dtype=dtype, device="npu")
    eps = 1e-6

    res = fused_rmsnorm_without_weight_golden(x, eps)
    ans = fused_rmsnorm_without_weight(x, eps)

    np.testing.assert_allclose(
        res.cpu().numpy(),
        ans.cpu().numpy(),
        rtol=1e-3,
        err_msg=f"Failed!",
    )

    print(f"Passed!")


if __name__ == "__main__":
    test_fused_scale()
