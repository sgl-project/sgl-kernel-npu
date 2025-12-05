import numpy as np
import torch
from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias


def add_rmsnorm_bias_quant_golden(
    input,
    residual,
    norm_weight,
    norm_bias,
    eps,
    quant_scale=None,
    quant_offset=None,
):
    input = input.to(torch.float32).cpu().numpy()
    residual = residual.to(torch.float32).cpu().numpy()
    norm_weight = norm_weight.to(torch.float32).cpu().numpy()
    norm_bias = norm_bias.to(torch.float32).cpu().numpy()

    out2 = input + residual
    reciprocal_std = 1 / np.sqrt(np.mean(out2**2, axis=-1, keepdims=True) + eps)
    out1 = out2 * reciprocal_std * norm_weight + norm_bias
    if quant_scale is not None:
        quant_scale = quant_scale.to(torch.float32).cpu().numpy()
        quant_offset = quant_offset.to(torch.float32).cpu().numpy()
        out1 = out1 * quant_scale + quant_offset
        out1 = np.round(out1)

    return out1, out2


def test_add_rmsnorm_bias():
    hidden_size = 6144
    input = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    residual = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    weight = torch.randn(hidden_size).to(torch.bfloat16).npu()
    bias = torch.randn(hidden_size).to(torch.bfloat16).npu()
    res1, res2 = add_rmsnorm_bias(input, residual, weight, bias, 1e-6)
    ans1, ans2 = add_rmsnorm_bias_quant_golden(input, residual, weight, bias, 1e-6)

    assert (
        np.testing.assert_allclose(
            res1.to(torch.float32).cpu().numpy(),
            ans1,
            rtol=5e-3,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            res2.to(torch.float32).cpu().numpy(),
            ans2,
            rtol=5e-3,
        )
        is None
    )

    # enable quant
    hidden_size = 6144
    input = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    residual = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    weight = torch.randn(hidden_size).to(torch.bfloat16).npu()
    bias = torch.randn(hidden_size).to(torch.bfloat16).npu()
    quant_scale = torch.randn(hidden_size).to(torch.bfloat16).npu()
    quant_offset = torch.randn(hidden_size).to(torch.bfloat16).npu()
    res1, res2 = add_rmsnorm_bias(
        input, residual, weight, bias, 1e-6, quant_scale, quant_offset
    )
    ans1, ans2 = add_rmsnorm_bias_quant_golden(
        input, residual, weight, bias, 1e-6, quant_scale, quant_offset
    )

    diff = res1.to(torch.float32).cpu().numpy() - ans1

    assert (diff <= 1).any()

    assert (
        np.testing.assert_allclose(
            res2.to(torch.float32).cpu().numpy(),
            ans2,
            rtol=5e-3,
        )
        is None
    )


if __name__ == "__main__":
    test_add_rmsnorm_bias()
