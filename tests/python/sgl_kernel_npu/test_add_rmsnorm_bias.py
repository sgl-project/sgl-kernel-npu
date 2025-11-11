import numpy as np
import torch
import torch_npu
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
    out2 = input + residual
    out1 = torch_npu.npu_rms_norm(out2, norm_weight, eps)[0] + norm_bias
    if quant_scale is not None:
        out1 = torch_npu.npu_quantize(
                    out1,
                    quant_scale,
                    quant_offset,
                    torch.qint8,
                    -1,
                    False,
                )
    return out1, out2


def test_add_rmsnorm_bias():
    hidden_size = 6144
    input = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    residual = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    weight = torch.randn(hidden_size).to(torch.bfloat16).npu()
    bias = torch.randn(hidden_size).to(torch.bfloat16).npu()
    res1, res2 = add_rmsnorm_bias(input, residual, weight, bias, 1e-6)
    ans1, ans2 = add_rmsnorm_bias_quant_golden(input, residual, weight, bias, 1e-6)

    assert torch.allclose(res1, ans1, rtol=5e-3)

    assert torch.allclose(res2, ans2, rtol=5e-3)

    # enable quant
    hidden_size = 6144
    input = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    residual = torch.randn(3, hidden_size).to(torch.bfloat16).npu()
    weight = torch.randn(hidden_size).to(torch.bfloat16).npu()
    bias = torch.randn(hidden_size).to(torch.bfloat16).npu()
    quant_scale = torch.randn(hidden_size).to(torch.bfloat16).npu()
    quant_offset = torch.randn(hidden_size).to(torch.bfloat16).npu()
    res1, res2 = add_rmsnorm_bias(input, residual, weight, bias, 1e-6, quant_scale, quant_offset)
    ans1, ans2 = add_rmsnorm_bias_quant_golden(input, residual, weight, bias, 1e-6, quant_scale, quant_offset)

    diff = res1 - ans1
    max_diff = torch.max(torch.abs(diff))
    assert max_diff <= 1

    assert torch.allclose(res2, ans2, rtol=5e-3)


if __name__ == "__main__":
    test_add_rmsnorm_bias()