"""Explicit Ascend C tests for public hc_combine: projections, out buffers and guards."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
F = torch.nn.functional

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import require_distinct_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

BF16_TIGHT = {"torch.bfloat16": AccuracyStandard(atol=0.0, rtol=8e-3)}
BF16_VS_FP64_REF = {"torch.bfloat16": AccuracyStandard(atol=0.02, rtol=0.02)}

perf_shape = [
    (32, 4, 2048),
    (2048, 4, 2048),
]

GATE_MODES = [
    "normal",  # Projected injection logits spanning both sigmoid regions.
    "saturated",  # Large projection gain: gates saturate to 0 or 2.
]


def _bf16_staged_reference(block, residual, normed, weight, hc, hs):
    """PyTorch reference matching ascendc's BF16 projection boundary."""
    rows = residual.shape[0]
    if rows == 0:
        return torch.empty_like(residual)
    gate_logits = F.linear(normed, weight)
    gate = 2.0 / (1.0 + torch.exp(-(gate_logits.float() / hc)))
    injection = block.float().unsqueeze(1) * gate.unsqueeze(-1)
    combined = residual.reshape(rows, hc, hs).float() + injection
    return combined.reshape(rows, hc * hs).to(torch.bfloat16)


def generate_data(shape, gate_mode, seed=11042026):
    """Create NPU inputs using the device context supplied by conftest."""
    rows, hc, hs = shape
    total_width = hc * hs
    torch.npu.manual_seed(seed)
    block = torch.randn(rows, hs, dtype=torch.bfloat16)
    residual = torch.randn(rows, total_width, dtype=torch.bfloat16)
    normed = torch.randn(rows, total_width, dtype=torch.bfloat16)
    if gate_mode == "normal":
        gain = 4.0
    elif gate_mode == "saturated":
        gain = 200.0
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    scale = gain * float(hc) * (total_width**-0.5)
    weight = (torch.randn(hc, total_width, dtype=torch.float32) * scale).to(
        torch.bfloat16
    )
    return block, residual, normed, weight


@pytest.mark.parametrize("gate_mode", GATE_MODES)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 4, 2048), id="qwen38-flash-next-decode-32x4x2048"),
        pytest.param((64, 4, 2560), id="qwen38-24l-decode-64x4x2560"),
        pytest.param((2048, 4, 2048), id="qwen38-flash-next-prefill-2048x4x2048"),
        pytest.param((1, 4, 2048), id="qwen38-flash-next-single-row"),
        pytest.param((16, 2, 2048), id="two-branches"),
        pytest.param((1, 1, 16), id="edge-minimum"),
        pytest.param((7, 2, 48), id="edge-odd-rows-narrow-width"),
        pytest.param((3, 8, 4096), id="edge-maximum-branches-and-width"),
        pytest.param((0, 4, 2048), id="edge-zero-rows"),
    ],
)
def test_hc_combine_ascendc(shape, gate_mode):
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    require_distinct_impls(hc_combine, "ascendc", "torch_native")
    _, hc, hs = shape
    inputs = generate_data(shape, gate_mode)
    executor = check_accuracy(
        hc_combine,
        args=(*inputs, hc, hs),
        impl="ascendc",
        impl_ref="torch_native",
        standard=BF16_VS_FP64_REF,
    )
    assert_accuracy(executor(), _bf16_staged_reference(*inputs, hc, hs), BF16_TIGHT)
    check_performance(
        executor, enable_perf=shape in perf_shape and gate_mode == "normal"
    )


def test_hc_combine_ascendc_graph_replay():
    """Captured Ascend C hc_combine graph must replay accurately after mutations."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    require_distinct_impls(hc_combine, "ascendc", "torch_native")
    block, residual, normed, weight = generate_data((16, 4, 2560), "normal")
    for _ in range(3):
        hc_combine(block, residual, normed, weight, 4, 2560, impl="ascendc")
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = hc_combine(block, residual, normed, weight, 4, 2560, impl="ascendc")
    block.mul_(-0.5)
    residual.add_(0.03125)
    graph.replay()
    torch.npu.synchronize()
    expected = _bf16_staged_reference(block, residual, normed, weight, 4, 2560)
    assert_accuracy(out, expected, BF16_TIGHT)
    assert torch.equal(
        out, hc_combine(block, residual, normed, weight, 4, 2560, impl="ascendc")
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 4, 2048), id="decode-32x4x2048"),
        pytest.param((1, 1, 16), id="edge-minimum"),
    ],
)
def test_hc_combine_leaves_inputs_unchanged(shape):
    """The residual is returned as a new tensor, not updated in place."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    require_test_impl(hc_combine, "ascendc")
    _, hc, hs = shape
    block, residual, normed, weight = generate_data(shape, "normal")
    before = residual.clone()
    result = hc_combine(block, residual, normed, weight, hc, hs, impl="ascendc")
    assert result.data_ptr() != residual.data_ptr()
    assert torch.equal(residual, before)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["ascendc", "torch_native"])
def test_hc_combine_out_buffer(impl):
    """Pre-allocated out buffers are written in place without mutating inputs."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    require_test_impl(hc_combine, impl)
    shape = (16, 4, 2048)
    _, hc, hs = shape
    block, residual, normed, weight = generate_data(shape, "normal")
    baseline = hc_combine(block, residual, normed, weight, hc, hs, impl=impl)

    out_kw = torch.full_like(residual, -1.0)
    ptr_kw = out_kw.data_ptr()
    ret_kw = hc_combine(block, residual, normed, weight, hc, hs, out=out_kw, impl=impl)
    assert ret_kw is out_kw
    assert ret_kw.data_ptr() == ptr_kw
    assert torch.equal(out_kw, baseline)

    out_pos = torch.full_like(residual, -1.0)
    ptr_pos = out_pos.data_ptr()
    ret_pos = hc_combine(block, residual, normed, weight, hc, hs, out_pos, impl=impl)
    assert ret_pos is out_pos
    assert ret_pos.data_ptr() == ptr_pos
    assert torch.equal(out_pos, baseline)

    n = residual.numel()
    pool = torch.empty(2 * n, dtype=torch.bfloat16, device=residual.device)
    res_slice = pool[:n].view_as(residual)
    res_slice.copy_(residual)
    out_slice = pool[n : 2 * n].view_as(residual)
    out_slice.fill_(-1.0)
    ptr_slice = out_slice.data_ptr()
    ret_slice = hc_combine(
        block, res_slice, normed, weight, hc, hs, out=out_slice, impl=impl
    )
    assert ret_slice is out_slice
    assert ret_slice.data_ptr() == ptr_slice
    assert torch.equal(out_slice, baseline)
    assert torch.equal(res_slice, residual)

    block0, res0, normed0, weight0 = generate_data((0, hc, hs), "normal")
    out0 = torch.empty((0, hc * hs), dtype=torch.bfloat16, device=residual.device)
    ret0 = hc_combine(block0, res0, normed0, weight0, hc, hs, out=out0, impl=impl)
    assert ret0 is out0
    assert ret0.shape == (0, hc * hs)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,message",
    [
        pytest.param("non_tensor", "torch.Tensor", id="rejects-non-tensor-out"),
        pytest.param("dtype", "BF16", id="rejects-float32-out"),
        pytest.param("cross_device", "same NPU", id="rejects-cpu-out"),
        pytest.param("shape", "same shape", id="rejects-wrong-shape-out"),
        pytest.param("strided", "contiguous", id="rejects-strided-out"),
        pytest.param("grad", "inference only", id="rejects-requires-grad-out"),
        pytest.param(
            "alias_residual", "alias or overlap", id="rejects-out-aliasing-residual"
        ),
        pytest.param(
            "alias_normed", "alias or overlap", id="rejects-out-aliasing-normed"
        ),
        pytest.param(
            "partial_overlap", "alias or overlap", id="rejects-partial-overlap-out"
        ),
        pytest.param(
            "zero_row_alias", "alias or overlap", id="rejects-zero-row-self-alias"
        ),
    ],
)
def test_hc_combine_rejects_invalid_out(case, message):
    """Invalid out buffers must raise ValueError before any kernel launch."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    block, residual, normed, weight = generate_data((8, 4, 2048), "normal")
    hc, hs = 4, 2048
    if case == "non_tensor":
        out = "bad"
    elif case == "dtype":
        out = torch.empty_like(residual, dtype=torch.float32)
    elif case == "cross_device":
        out = torch.empty_like(residual, device="cpu")
    elif case == "shape":
        out = torch.empty((8, 2048), dtype=torch.bfloat16, device=residual.device)
    elif case == "strided":
        out = torch.empty(
            (8, 2 * 4 * 2048), dtype=torch.bfloat16, device=residual.device
        )[:, : 4 * 2048]
    elif case == "grad":
        out = torch.empty_like(residual).requires_grad_(True)
    elif case == "alias_residual":
        out = residual
    elif case == "alias_normed":
        out = normed
    elif case == "partial_overlap":
        n = residual.numel()
        pool = torch.empty(n + n // 2, dtype=torch.bfloat16, device=residual.device)
        residual = pool[:n].view_as(residual)
        out = pool[n // 2 : n + n // 2].view_as(residual)
    elif case == "zero_row_alias":
        block, residual, normed, weight = generate_data((0, 4, 2048), "normal")
        out = residual
    with pytest.raises(ValueError, match=message):
        hc_combine(block, residual, normed, weight, hc, hs, out=out)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,message",
    [
        pytest.param("dtype", "BF16", id="rejects-float16"),
        pytest.param("device", "same NPU", id="rejects-cpu"),
        pytest.param("cross_device", "same NPU", id="rejects-cross-device"),
        pytest.param("rank", "2-D", id="rejects-3d"),
        pytest.param("bool_hc", "hc", id="rejects-bool-hc-count"),
        pytest.param("hc_count", "power of two", id="rejects-three-branches"),
        pytest.param("too_many", "power of two", id="rejects-sixteen-branches"),
        pytest.param("bool_hidden", "hs", id="rejects-bool-hidden-size"),
        pytest.param("hidden_size", "multiple of 16", id="rejects-unaligned-hidden"),
        pytest.param(
            "residual_shape", "residual must have shape", id="rejects-residual"
        ),
        pytest.param("normed_shape", "normed must have shape", id="rejects-normed"),
        pytest.param("block_shape", "block must have shape", id="rejects-block"),
        pytest.param("inject_shape", "weight must have shape", id="rejects-inject"),
        pytest.param("strided", "contiguous", id="rejects-strided-residual"),
        pytest.param("grad", "inference only", id="rejects-requires-grad"),
    ],
)
def test_hc_combine_rejects_inputs(case, message):
    """Unsupported inputs must raise ValueError before any implementation runs."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    block, residual, normed, weight = generate_data((8, 4, 2048), "normal")
    hc, hs = 4, 2048
    if case == "dtype":
        block = block.to(torch.float16)
    elif case == "device":
        block, residual, normed, weight = (
            block.cpu(),
            residual.cpu(),
            normed.cpu(),
            weight.cpu(),
        )
    elif case == "cross_device":
        weight = weight.cpu()
    elif case == "rank":
        residual = residual.reshape(8, 4, 2048)
    elif case == "bool_hc":
        hc = True
    elif case == "hc_count":
        hc = 3
    elif case == "too_many":
        hc = 16
    elif case == "bool_hidden":
        hs = True
    elif case == "hidden_size":
        hs = 24
    elif case == "residual_shape":
        residual = torch.randn(8, 2048, dtype=torch.bfloat16)
    elif case == "normed_shape":
        normed = torch.randn(8, 2048, dtype=torch.bfloat16)
    elif case == "block_shape":
        block = torch.randn(8, 1024, dtype=torch.bfloat16)
    elif case == "inject_shape":
        weight = torch.randn(3, 4 * 2048, dtype=torch.bfloat16)
    elif case == "strided":
        residual = torch.randn(8, 2 * 4 * 2048, dtype=torch.bfloat16)[:, : 4 * 2048]
    elif case == "grad":
        residual = residual.requires_grad_()
    with pytest.raises(ValueError, match=message):
        hc_combine(block, residual, normed, weight, hc, hs)
