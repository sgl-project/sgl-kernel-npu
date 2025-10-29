import numpy as np
import torch
import torch_npu
from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

def test_swiglu_quant():
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()
    # create inputs
    s, h = 4096, 4096
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = torch.Tensor([0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]).npu().to(torch.int64)
    # torch native
    swglu_out = torch_npu.npu_swiglu(x)
    ans1, ans2 = torch_npu.npu_dynamic_quant(swglu_out)
    # fused_triton_kernel
    res1, res2 = swiglu_quant(x, group_list, group_list_type=1)

    real_tokens = torch.sum(group_list)
    diff = res1[:real_tokens, :] - ans1[:real_tokens, :]
    max_diff = torch.max(torch.abs(diff))
    assert max_diff <= 1

    diff_rate = torch.sum(torch.abs(diff)) / (real_tokens * h // 2)
    assert diff_rate < 2e-2

    assert np.testing.assert_allclose(
        to_numpy(res2[:real_tokens]),
        to_numpy(ans2[:real_tokens]),
        rtol=5e-3,
    ) is None


if __name__ == "__main__":
    test_swiglu_quant()
