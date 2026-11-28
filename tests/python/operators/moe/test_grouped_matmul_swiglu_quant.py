"""A3 fused GMM accuracy and changed-input graph replay through the public API."""

import pytest
import sgl_kernel_npu
import torch
import torch_npu
from sgl_kernel_npu.operators.moe import grouped_matmul_swiglu_quant

pytestmark = pytest.mark.skipif(
    not any(
        "sgl_grouped_matmul_swiglu_quant" in item["ops"]
        for item in getattr(sgl_kernel_npu, "_manifest", {}).get("third_party", ())
    ),
    reason="GroupedMatmulSwigluQuant wheel required",
)


@pytest.fixture
def internal_format():
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    yield
    torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": previous.decode()})


@pytest.mark.parametrize("counts,limit", [([0, 1, 5, 0], 10.0), ([0, 0, 0, 0], 0.0)])
def test_grouped_matmul_swiglu_quant(limit, counts, internal_format):
    torch.manual_seed(17)
    x = torch.randint(-100, 100, (32, 512), dtype=torch.int8, device="npu")
    weight = torch.randint(-100, 100, (4, 1024, 512), dtype=torch.int8, device="npu")
    weight = torch_npu.npu_format_cast(weight.transpose(1, 2).contiguous(), 29)
    scale = torch.full((4, 1024), 0.01, device="npu", dtype=torch.bfloat16)
    x_scale = torch.full((32,), 0.01, device="npu")
    ends = torch.tensor(counts, device="npu", dtype=torch.int64).cumsum(0)
    args = (x, weight, scale, x_scale, ends, limit)
    grouped_matmul_swiglu_quant(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = grouped_matmul_swiglu_quant(*args)
    for new_counts in ([8] * 4, [0] * 4, [0, 1, 5, 0]):
        x.random_(-100, 100)
        ends.copy_(torch.tensor(new_counts, device="npu").cumsum(0))
        graph.replay()
        expected = grouped_matmul_swiglu_quant(*args, impl="torch_native")
        valid = sum(new_counts)
        if valid:
            a, b = [q[:valid].float() * s[:valid, None] for q, s in (actual, expected)]
            relative_rms = (
                (a - b).square().mean() / b.square().mean().clamp_min(1e-30)
            ).sqrt()
            assert relative_rms.item() < 0.01


@pytest.mark.parametrize("scale_groups", [1, 2])
def test_grouped_matmul_swiglu_quant_w4a8(scale_groups, internal_format):
    torch.manual_seed(31)
    experts, rows, k, n = 4, 32, 512, 1024
    x = torch.randint(-96, 96, (rows, k), dtype=torch.int8, device="npu")
    unpacked = torch.randint(-8, 8, (experts, k, n), dtype=torch.int8, device="npu")
    packed = (unpacked[..., ::2] & 15) | (unpacked[..., 1::2] << 4)
    weight = torch_npu.npu_format_cast(packed.contiguous(), 29)
    weight = weight.view(torch.int32).contiguous()
    assert torch.ops.npu.get_npu_format(weight) == 29

    scales = torch.randint(1, 4, (experts, scale_groups, n), device="npu").float()
    scales *= 1 / 256
    bias = (
        unpacked.float().reshape(experts, scale_groups, -1, n).sum(dim=2) * scales
    ).sum(dim=1) * 8
    scale_bits = scales.view(torch.int32).to(torch.int64)
    if scale_groups == 1:
        scale_bits = scale_bits.squeeze(1)
    x_scale = torch.full((rows,), 1 / 32, device="npu")
    ends = torch.tensor([0, 3, 8, 8], device="npu", dtype=torch.int64)
    args = (x, weight, scale_bits, x_scale, ends, 0.25)
    kwargs = {"weight_assist_matrix": bias}

    grouped_matmul_swiglu_quant(*args, **kwargs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        quant, scale = grouped_matmul_swiglu_quant(*args, **kwargs)
    assert quant.shape == (rows, n // 2) and quant.dtype == torch.int8
    assert scale.shape == (rows,) and scale.dtype == torch.float32

    for new_counts in ([8] * 4, [0] * 4, [0, 1, 5, 0]):
        x.random_(-96, 96)
        ends.copy_(torch.tensor(new_counts, device="npu").cumsum(0))
        graph.replay()
        valid = sum(new_counts)
        if valid:
            ref_quant, ref_scale = grouped_matmul_swiglu_quant(
                *args, **kwargs, impl="torch_native"
            )
            actual = quant[:valid].float() * scale[:valid, None]
            expected = ref_quant[:valid].float() * ref_scale[:valid, None]
            relative_rms = (
                (actual - expected).square().mean()
                / expected.square().mean().clamp_min(1e-30)
            ).sqrt()
            assert relative_rms.item() < 0.02
