import torch
import torch_npu

from sgl_kernel_npu.activation.situ import situ_and_mul, situ_and_mul_quant


def test_situ_and_mul_quant_matches_materialized_bf16():
    torch.manual_seed(42)
    for rows, half_cols in ((8, 768), (16, 384)):
        x = torch.randn(
            (rows, 2 * half_cols), dtype=torch.bfloat16, device="npu"
        )
        activated = situ_and_mul(x, beta=4.0, linear_beta=25.0)
        expected, expected_scale = torch_npu.npu_dynamic_quant(activated)

        actual, actual_scale = situ_and_mul_quant(
            x,
            beta=4.0,
            linear_beta=25.0,
            need_quant=True,
        )

        torch.testing.assert_close(
            actual_scale, expected_scale.flatten(), rtol=1e-6, atol=1e-8
        )
        quant_error = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
        assert quant_error.max().item() <= 1
        assert (quant_error != 0).float().mean().item() <= 1e-3
