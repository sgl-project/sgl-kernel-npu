import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant


def swiglu_silu_clamp_mul_native(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """Fp32 clamped SwiGLU reference, matching the fused kernel's do_limit path."""
    gate, up = x.to(torch.float32).chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    out = gate * up
    return out


def quantize_symmetric(x: torch.Tensor, max_val: int = 127):
    """Symmetric int8 quantization matching the fused kernel (scale = max / 127)."""
    scale = torch.amax(torch.abs(x), dim=-1) / max_val
    out = torch.floor(x / scale.unsqueeze(-1) + 0.5)
    out = torch.clamp(out, -max_val, max_val).to(torch.int8)
    return out, scale


def test_swiglu_quant():
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    # create inputs
    s, h = 4096, 3072
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = (
        torch.Tensor([0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0])
        .npu()
        .to(torch.int64)
    )
    # torch native: match the fused kernel's fp32 SwiGLU and quantization path
    gate, up = x.to(torch.float32).chunk(2, dim=-1)
    swglu_out = gate * torch.sigmoid(gate) * up
    ans1, ans2 = quantize_symmetric(swglu_out)
    # fused_triton_kernel
    res1, res2 = swiglu_quant(x, group_list, group_list_type=1)

    real_tokens = torch.sum(group_list)
    # Compare in float32: int8 - int8 wraps around (e.g. -128 - 127 == 1) and
    # could make the diff look small even on large errors.
    diff = res1[:real_tokens, :].float() - ans1[:real_tokens, :].float()

    max_diff = torch.max(torch.abs(diff))
    assert max_diff <= 1

    diff_rate = torch.sum(torch.abs(diff)) / (real_tokens * h // 2)
    assert diff_rate < 2e-2

    assert (
        np.testing.assert_allclose(
            to_numpy(res2[:real_tokens]),
            to_numpy(ans2[:real_tokens]),
            rtol=5e-3,
        )
        is None
    )


def test_swiglu_quant_with_limit():
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    # create inputs
    s, h = 4096, 3072
    x = torch.randn((s, h), dtype=torch.bfloat16).npu()
    group_list = (
        torch.Tensor([0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0])
        .npu()
        .to(torch.int64)
    )
    # torch native: match the fused kernel's fp32 clamped SwiGLU + symmetric quant
    swglu_out = swiglu_silu_clamp_mul_native(x)
    ans1, ans2 = quantize_symmetric(swglu_out)
    # fused_triton_kernel
    res1, res2 = swiglu_quant(x, group_list, group_list_type=1, do_limit=True)

    real_tokens = torch.sum(group_list)
    # Compare in float32: int8 - int8 wraps around (e.g. -128 - 127 == 1)
    diff = res1[:real_tokens, :].float() - ans1[:real_tokens, :].float()

    max_diff = torch.max(torch.abs(diff))
    assert max_diff <= 1

    diff_rate = torch.sum(torch.abs(diff)) / (real_tokens * h // 2)
    assert diff_rate < 2e-2

    assert (
        np.testing.assert_allclose(
            to_numpy(res2[:real_tokens]),
            to_numpy(ans2[:real_tokens]),
            rtol=5e-3,
        )
        is None
    )


if __name__ == "__main__":
    test_swiglu_quant()
    test_swiglu_quant_with_limit()
