"""Compare the public quantized matmuls against independent PyTorch calculations."""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def require_arch35():
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH35:
        pytest.skip("This quantized matmul workload requires arch35")


@pytest.mark.parametrize(
    "m",
    [
        pytest.param(1, id="decode"),
        pytest.param(17, id="tail", marks=pytest.mark.extra_checks),
    ],
)
def test_quant_matmul_int8_with_pertoken_scale(m):
    from sgl_kernel_npu.operators.matmul import quant_matmul

    k, n = 128, 64
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randint(
        -3, 4, (m, k), dtype=torch.int8, device="cpu", generator=generator
    )
    rhs = torch.randint(
        -3, 4, (k, n), dtype=torch.int8, device="cpu", generator=generator
    )
    channel_scale = torch.full((n,), 0.25, dtype=torch.float32, device="cpu")
    token_scale = torch.full((m,), 0.125, dtype=torch.float32, device="cpu")
    expected = (lhs.float() @ rhs.float()) * channel_scale * token_scale[:, None]
    actual = quant_matmul(
        lhs.npu(),
        rhs.npu(),
        channel_scale.npu(),
        pertoken_scale=token_scale.npu(),
        output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(actual.cpu(), expected.to(torch.bfloat16))


@pytest.mark.parametrize(
    "shape", [pytest.param((1, 320, 256), id="a16w8-per-channel-decode")]
)
def test_weight_quant_batchmatmul_int8(shape):
    from sgl_kernel_npu.operators.matmul import weight_quant_batchmatmul

    # The vendor A16W8 ND example uses K=320, N=256. Use one row for decode.
    m, k, n = shape
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randn((m, k), generator=generator, device="cpu", dtype=torch.float16)
    weights = torch.randint(
        -8, 8, (k, n), generator=generator, device="cpu", dtype=torch.int8
    )
    scales = (torch.arange(n, device="cpu") % 4 + 1).to(torch.float16) / 16
    offsets = (torch.arange(n, device="cpu") % 3 - 1).to(torch.float16)
    scales, offsets = scales.unsqueeze(0), offsets.unsqueeze(0)
    expected = lhs.float() @ ((weights.float() + offsets) * scales)

    actual = weight_quant_batchmatmul(
        lhs.npu(), weights.npu(), scales.npu(), antiquant_offset=offsets.npu()
    )

    torch.testing.assert_close(
        actual.cpu(), expected.to(torch.float16), rtol=0.002, atol=0.002
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((2, 2, 512, 128), id="deepseek-v4-mxfp8-wo-a-layout")]
)
def test_transpose_quant_batchmatmul_mxfp8(require_arch35, shape):
    from sgl_kernel_npu.operators.matmul import transpose_quant_batchmatmul

    # Model the grouped WoA permutation used in SGLang DeepSeek-V4, with
    # reduced dimensions. E8M0 input scale exponent bytes have bias 127.
    m, batch, k, n = shape
    generator = torch.Generator(device="cpu").manual_seed(7)
    lhs = torch.randint(-3, 4, (m, batch, k), generator=generator, device="cpu")
    rhs = torch.randint(-3, 4, (batch, k, n), generator=generator, device="cpu")
    lhs_exponents = torch.randint(
        -2, 1, (m, batch, k // 32), generator=generator, device="cpu"
    )
    rhs_exponents = torch.randint(
        -2, 1, (batch, k // 32, n), generator=generator, device="cpu"
    )
    x1_scale = (lhs_exponents + 127).to(torch.uint8).reshape(m, batch, k // 64, 2)
    x2_scale = (rhs_exponents + 127).to(torch.uint8)
    x2_scale = x2_scale.reshape(batch, k // 64, 2, n).transpose(-1, -2).contiguous()
    dequant_lhs = lhs * (2.0**lhs_exponents).repeat_interleave(32, dim=-1)
    dequant_rhs = rhs * (2.0**rhs_exponents).repeat_interleave(32, dim=1)
    expected = (dequant_lhs.transpose(0, 1) @ dequant_rhs).transpose(0, 1)

    actual = transpose_quant_batchmatmul(
        lhs.to(torch.float8_e4m3fn).npu(),
        rhs.to(torch.float8_e4m3fn).npu(),
        torch.bfloat16,
        x1_scale=x1_scale.npu().view(torch.float8_e8m0fnu),
        x2_scale=x2_scale.npu().view(torch.float8_e8m0fnu),
        group_sizes=(0, 0, 32),
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )

    torch.testing.assert_close(actual.cpu(), expected.to(torch.bfloat16))


@pytest.mark.parametrize(
    "shape", [pytest.param((1, 4096, 128), id="qwen3-w4a4-linear-decode-tile")]
)
def test_dual_level_quant_matmul_w4a4(require_arch35, shape):
    import torch_npu
    from sgl_kernel_npu.operators.matmul import dual_level_quant_matmul
    from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant

    # Follow SGLang NPUDualLevelMXFP4LinearMethod: one activation row,
    # full hidden width, reduced output width and FRACTAL_NZ packed weight.
    # Each group contains +/-6, so all inputs are exactly representable
    # with both scales equal to one and the unquantized GEMM is a valid golden.
    m, k, n = shape
    lhs = torch.tensor([6, 3, -6, -3], dtype=torch.bfloat16, device="cpu")
    lhs = lhs.repeat(m, k // 4)
    weight_values = torch.tensor([6, -3, 3, -6], dtype=torch.bfloat16, device="cpu")
    weights = weight_values.repeat(n, k // 4)
    weights[1::2].neg_()
    expected = lhs.float() @ weights.float().transpose(0, 1)

    x1, x1_l0, x1_l1 = dynamic_dual_level_mx_quant(lhs.npu())
    x2, x2_l0, x2_l1 = dynamic_dual_level_mx_quant(weights.npu())
    x2_l0 = x2_l0.reshape(n, k // 512).transpose(0, 1).contiguous()
    fractal_nz = 29
    internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT").decode()
    try:
        torch_npu.npu.config.allow_internal_format = True
        weights_nz = torch_npu.npu_format_cast(
            x2.view(torch.int8), fractal_nz, customize_dtype=torch.int8
        )
        actual = dual_level_quant_matmul(
            x1, weights_nz, x1_l0, x2_l0, x1_l1, x2_l1, output_dtype=torch.bfloat16
        ).cpu()
    finally:
        torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": internal_format})

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))
