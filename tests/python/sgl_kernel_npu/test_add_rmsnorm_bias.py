import numpy as np
import torch
from sgl_kernel_npu.norm.add_rmsnorm_bias import add_gemma_rms_norm, add_rmsnorm_bias


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
    res1, res2 = add_rmsnorm_bias(
        input,
        residual,
        weight,
        1e-6,
        norm_bias=bias,
        quant_scale=None,
        quant_offset=None,
    )
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
        input,
        residual,
        weight,
        1e-6,
        norm_bias=bias,
        quant_scale=quant_scale,
        quant_offset=quant_offset,
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


def reference_add_gemma_rms_norm(hidden_state, weight, residual, variance_epsilon):
    # Step 1: Add
    add_output = hidden_state + residual

    # Step 2: RMS Norm (Gemma style: x * (w + 1) / sqrt(mean(x^2) + eps))
    dtype = add_output.dtype
    add_output_fp32 = add_output.to(torch.float32)
    variance = torch.mean(add_output_fp32**2, dim=-1, keepdim=True)
    norm_output_fp32 = add_output_fp32 * torch.rsqrt(variance + variance_epsilon)
    norm_output_fp32 = norm_output_fp32 * (weight.to(torch.float32) + 1.0)
    norm_output = norm_output_fp32.to(dtype)

    return norm_output, add_output


def test_add_gemma_rms_norm():
    torch.manual_seed(0)
    device = torch.device("npu")

    test_cases = [
        (8, 512),
        (16, 1024),
        (32, 2048),
        (1, 256),
    ]

    variance_epsilon = 1e-6

    for batch, dim in test_cases:
        print(f"Testing batch={batch}, dim={dim}")

        hidden_state = torch.randn(batch, dim, device=device, dtype=torch.float16)
        residual = torch.randn(batch, dim, device=device, dtype=torch.float16)
        weight = torch.randn(dim, device=device, dtype=torch.float16)

        # Triton output
        norm_out_triton, add_out_triton = add_gemma_rms_norm(
            hidden_state, weight, residual, variance_epsilon
        )

        # Reference output
        norm_out_ref, add_out_ref = reference_add_gemma_rms_norm(
            hidden_state, weight, residual, variance_epsilon
        )

        # Compare
        assert torch.allclose(add_out_triton, add_out_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(norm_out_triton, norm_out_ref, atol=1e-2, rtol=1e-2)

    print("All tests passed!")


if __name__ == "__main__":
    test_add_rmsnorm_bias()
    test_add_gemma_rms_norm()
