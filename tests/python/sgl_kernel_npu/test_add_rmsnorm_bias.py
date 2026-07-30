import numpy as np
import pytest
import torch
import torch_npu
from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias
from sgl_kernel_npu.norm.gemma_rmsnorm import (
    add_gemma_rms_norm,
    gemma_rms_norm,
)


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
    add_output = hidden_state if residual is None else hidden_state + residual

    # Step 2: RMS Norm (Gemma style: x * (w + 1) / sqrt(mean(x^2) + eps))
    dtype = add_output.dtype
    add_output_fp32 = add_output.to(torch.float32)
    variance = torch.mean(add_output_fp32**2, dim=-1, keepdim=True)
    norm_output_fp32 = add_output_fp32 * torch.rsqrt(variance + variance_epsilon)
    norm_output_fp32 = norm_output_fp32 * (weight.to(torch.float32) + 1.0)
    norm_output = norm_output_fp32.to(dtype)

    return norm_output, add_output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (3, 256),
        (2, 3, 2048),
        (2, 4096),
        (1, 5120),
    ],
)
@pytest.mark.parametrize("has_residual", [False, True])
def test_add_gemma_rms_norm(shape, dtype, has_residual):
    torch.manual_seed(0)
    device = torch.device("npu")
    variance_epsilon = 1e-6
    hidden_state = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype) if has_residual else None
    weight = torch.randn(shape[-1], device=device, dtype=dtype)
    hidden_state_before = hidden_state.clone()
    residual_before = residual.clone() if residual is not None else None
    weight_before = weight.clone()

    if residual is None:
        norm_out_triton = gemma_rms_norm(hidden_state, weight, variance_epsilon)
        add_out_triton = hidden_state
    else:
        norm_out_triton, add_out_triton = add_gemma_rms_norm(
            hidden_state, weight, residual, variance_epsilon
        )
    norm_out_ref, add_out_ref = reference_add_gemma_rms_norm(
        hidden_state, weight, residual, variance_epsilon
    )

    assert norm_out_triton.shape == hidden_state.shape
    assert norm_out_triton.dtype == hidden_state.dtype
    assert add_out_triton.shape == hidden_state.shape
    assert add_out_triton.dtype == hidden_state.dtype
    torch.testing.assert_close(add_out_triton, add_out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(norm_out_triton, norm_out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(hidden_state, hidden_state_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
    if residual is None:
        assert add_out_triton.data_ptr() == hidden_state.data_ptr()
    else:
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_residual", [False, True])
def test_gemma_rms_norm_noncontiguous_input(dtype, has_residual):
    torch.manual_seed(0)
    device = torch.device("npu")
    eps = 1e-6
    hidden_size = 5120
    hidden_state = torch.randn(2, 3, hidden_size * 2, device=device, dtype=dtype)[
        ..., ::2
    ]
    residual = (
        torch.randn(2, 3, hidden_size * 2, device=device, dtype=dtype)[..., ::2]
        if has_residual
        else None
    )
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    hidden_state_before = hidden_state.clone()
    residual_before = residual.clone() if residual is not None else None
    weight_before = weight.clone()

    if residual is None:
        output = gemma_rms_norm(hidden_state, weight, eps)
    else:
        output, residual_sum = add_gemma_rms_norm(hidden_state, weight, residual, eps)
    reference, reference_sum = reference_add_gemma_rms_norm(
        hidden_state, weight, residual, eps
    )

    assert not hidden_state.is_contiguous()
    assert output.shape == hidden_state.shape
    assert output.dtype == hidden_state.dtype
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(hidden_state, hidden_state_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
    if residual is not None:
        assert not residual.is_contiguous()
        torch.testing.assert_close(residual_sum, reference_sum, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemma_rms_norm_matches_npu_rms_norm(dtype):
    torch.manual_seed(0)
    device = torch.device("npu")
    eps = 1e-6
    hidden_state = torch.randn(8, 4096, device=device, dtype=dtype)
    weight = torch.randn(4096, device=device, dtype=dtype)

    output = gemma_rms_norm(hidden_state, weight, eps)
    fallback = torch_npu.npu_rms_norm(hidden_state, 1.0 + weight, eps)[0]

    torch.testing.assert_close(output, fallback, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_add_gemma_rms_norm_matches_npu_rms_norm(dtype):
    torch.manual_seed(0)
    device = torch.device("npu")
    eps = 1e-6
    hidden_state = torch.randn(8, 4096, device=device, dtype=dtype)
    residual = torch.randn_like(hidden_state)
    weight = torch.randn(4096, device=device, dtype=dtype)

    output, residual_sum = add_gemma_rms_norm(hidden_state, weight, residual, eps)
    fallback_sum = hidden_state + residual
    fallback = torch_npu.npu_rms_norm(fallback_sum, 1.0 + weight, eps)[0]

    torch.testing.assert_close(residual_sum, fallback_sum, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(output, fallback, atol=2e-2, rtol=2e-2)


def test_gemma_rms_norm_empty_input():
    hidden_state = torch.empty(0, 5120, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(5120, device="npu", dtype=torch.bfloat16)

    output = gemma_rms_norm(hidden_state, weight, 1e-6)

    assert output.shape == hidden_state.shape
    assert output.dtype == hidden_state.dtype
    assert output.numel() == 0


def test_add_gemma_rms_norm_empty_input():
    hidden_state = torch.empty(0, 5120, device="npu", dtype=torch.bfloat16)
    residual = torch.empty_like(hidden_state)
    weight = torch.randn(5120, device="npu", dtype=torch.bfloat16)

    output, residual_sum = add_gemma_rms_norm(hidden_state, weight, residual, 1e-6)

    assert output.shape == hidden_state.shape
    assert residual_sum.shape == hidden_state.shape
    assert output.numel() == 0
    assert residual_sum.numel() == 0


def test_gemma_rms_norm_validates_contract():
    hidden_state = torch.randn(2, 256, device="npu", dtype=torch.bfloat16)
    residual = torch.randn_like(hidden_state)
    weight = torch.randn(256, device="npu", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="match input.shape"):
        gemma_rms_norm(hidden_state, weight[:-1])
    with pytest.raises(ValueError, match="same shape"):
        add_gemma_rms_norm(hidden_state, weight, residual[:, :-1])
    with pytest.raises(TypeError, match="torch.float16 or torch.bfloat16"):
        gemma_rms_norm(hidden_state.float(), weight.float())
    with pytest.raises(TypeError, match="same dtype"):
        add_gemma_rms_norm(hidden_state, weight, residual.half())
    with pytest.raises(ValueError, match="same device"):
        gemma_rms_norm(hidden_state, weight.cpu())


if __name__ == "__main__":
    test_add_rmsnorm_bias()
    test_add_gemma_rms_norm((3, 256), torch.float16, True)
