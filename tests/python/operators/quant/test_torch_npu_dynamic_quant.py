"""Numerical checks of the public TorchNPU dynamic quantization entry points."""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def require_arch35():
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH35:
        pytest.skip("The FP8 and dual-level MXFP4 workloads require arch35")


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="linear-decode-width"),
        pytest.param((3, 129), id="unaligned-tail", marks=pytest.mark.extra_checks),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dynamic_quant_int8(shape, dtype):
    from sgl_kernel_npu.operators.quant import dynamic_quant

    # Integer inputs with each row's maximum at 127 make the reference scale
    # exactly one and avoid differences at round-to-nearest boundaries.
    x = (torch.arange(shape[0] * shape[1], device="cpu").reshape(shape) % 255) - 127
    x[:, 0] = 127
    values, scales = dynamic_quant(x.to(device="npu", dtype=dtype), dst_type=torch.int8)

    torch.testing.assert_close(values.cpu(), x.to(torch.int8), atol=0, rtol=0)
    torch.testing.assert_close(
        scales.cpu(), torch.ones(shape[:-1], dtype=torch.float32, device="cpu")
    )


@pytest.mark.parametrize("shape", [pytest.param((4, 128), id="dsa-indexer-fp8-head")])
def test_dynamic_block_quant_fp8(require_arch35, shape):
    from sgl_kernel_npu.operators.quant import dynamic_block_quant

    # The DSA indexer uses 128-element row blocks. The chosen input gives
    # exact E4M3 values: scale = max(abs(block)) / 448.
    payload = torch.tensor([448, -224, 112, 0], dtype=torch.float32, device="cpu")
    expected = payload.repeat(shape[0], shape[1] // payload.numel())
    expected_scales = torch.tensor(
        [[0.125], [0.25], [0.5], [1.0]], dtype=torch.float32, device="cpu"
    )
    x = (expected * expected_scales).to(torch.bfloat16).npu()
    values, scales = dynamic_block_quant(
        x, dst_type=torch.float8_e4m3fn, row_block_size=1, col_block_size=128
    )

    assert values.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(values.cpu().float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(scales.cpu(), expected_scales)


@pytest.mark.parametrize(
    "shape", [pytest.param((2, 2, 512), id="deepseek-v4-wo-a-mxfp8")]
)
def test_dynamic_mx_quant_fp8(require_arch35, shape):
    from sgl_kernel_npu.operators.quant import dynamic_mx_quant

    # The grouped WoA projection uses one E8M0 scale per 32 elements.
    # In scale algorithm 0, exponent = floor(log2(max(abs(block)))) - 8.
    # Values selected here are exactly representable before and after scaling.
    blocks = shape[-1] // 32
    exponents = torch.arange(shape[0] * shape[1] * blocks, device="cpu")
    exponents = (exponents.reshape(*shape[:-1], blocks) % 4) - 8
    payload = torch.tensor([256, -128, 64, 0], dtype=torch.float32, device="cpu")
    expected = payload.repeat(*shape[:-1], blocks, 32 // payload.numel())
    x = (expected * (2.0**exponents).unsqueeze(-1)).reshape(shape)
    values, scales = dynamic_mx_quant(
        x.to(torch.bfloat16).npu(), dst_type=torch.float8_e4m3fn
    )

    assert values.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        values.cpu().float(), expected.reshape(shape), atol=0, rtol=0
    )
    expected_scales = (
        (exponents + 127).to(torch.uint8).reshape(*shape[:-1], blocks // 2, 2)
    )
    torch.testing.assert_close(
        scales.view(torch.uint8).cpu(), expected_scales, atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "shape", [pytest.param((1, 4096), id="qwen3-w4a4-linear-decode")]
)
def test_dynamic_dual_level_mx_quant_fp4(require_arch35, shape):
    from sgl_kernel_npu.operators.quant import dynamic_dual_level_mx_quant

    # Model one W4A4 activation row. Every 512-element group has a maximum of
    # 6 times the first scale. Within that group alternate 32-element blocks
    # with level-1 scales 1 and 0.5 to check both independently.
    rows, width = shape
    level0_blocks = width // 512
    expected_l0 = 2.0 ** (torch.arange(level0_blocks, device="cpu") % 4 - 2)
    expected_l0 = expected_l0.expand(rows, level0_blocks)
    level1_blocks = width // 32
    level1_exponents = -(torch.arange(level1_blocks, device="cpu") % 2)
    base = torch.tensor([6, 3, -6, -3], dtype=torch.float32, device="cpu").repeat(8)
    x = (
        expected_l0.repeat_interleave(16, dim=-1).unsqueeze(-1)
        * (2.0**level1_exponents).view(1, level1_blocks, 1)
        * base
    ).reshape(shape)

    packed, level0, level1 = dynamic_dual_level_mx_quant(
        x.to(torch.bfloat16).npu(), round_mode="rint"
    )

    # FP4 E2M1 stores 6/3/-6/-3 as nibbles 7/5/15/13. The lower nibble
    # encodes the first value in each consecutive pair.
    expected_packed = torch.tensor([0x57, 0xDF], dtype=torch.uint8, device="cpu")
    expected_packed = expected_packed.repeat(rows, width // 4)
    expected_l1 = (level1_exponents + 127).to(torch.uint8)
    expected_l1 = expected_l1.reshape(1, level1_blocks // 2, 2)
    expected_l1 = expected_l1.expand(rows, -1, -1)
    torch.testing.assert_close(
        packed.view(torch.uint8).cpu(), expected_packed, atol=0, rtol=0
    )
    torch.testing.assert_close(level0.cpu(), expected_l0, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        level1.view(torch.uint8).cpu(), expected_l1, atol=0, rtol=0
    )
