import numpy as np
import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm


def test_l1_norm():
    input = torch.randn(2048, 8).to(torch.bfloat16).npu()
    res = l1_norm(input)

    # gloden
    input = input.to(torch.float32).cpu().numpy()
    ans = input / input.sum(axis=-1, keepdims=True)

    assert (
        np.testing.assert_allclose(
            res.cpu().numpy(),
            ans,
            rtol=5e-3,
        )
        is None
    )


if __name__ == "__main__":
    test_l1_norm()
