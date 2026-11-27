"""Explicit Ascend C tests for public hc_mix: low-rank projections, NZ weights and guards."""

import importlib.util

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
F = torch.nn.functional

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import require_distinct_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Tight 1-2 BF16 ulp check against the BF16-projection staged reference, plus
# broader check against the CPU FP64-intermediate torch_native reference.
BF16_TIGHT = {"torch.bfloat16": AccuracyStandard(atol=0.0, rtol=8e-3)}
BF16_VS_FP64_REF = {"torch.bfloat16": AccuracyStandard(atol=0.04, rtol=0.04)}

FRACTAL_NZ = 29

perf_shape = [
    (32, 4, 2048, 320),
    (2048, 4, 2048, 320),
]

GATE_MODES = [
    "normal",  # Gate logits spanning the curved sigmoid region and moderate tails.
    "saturated",  # Large projection gain: gates saturate cleanly to 0 or 1.
]

UP_LAYOUTS = [
    "ND",  # Standard row-major [hc * hs, low_rank] weight.
    "FRACTAL_NZ",  # Pre-converted FRACTAL_NZ weight for faster decode GEMM.
]


def _cast_fractal_nz(weight):
    """Convert a 2-D weight to FRACTAL_NZ without mutating global NPU options."""
    if importlib.util.find_spec("tbe") is None:
        pytest.skip("FRACTAL_NZ cast requires CANN tbe on PYTHONPATH")
    old_opt = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    old_allow = old_opt is not None and old_opt.decode() == "enable"
    try:
        torch.npu.config.allow_internal_format = True
        nz = torch_npu.npu_format_cast(weight.contiguous(), FRACTAL_NZ)
    finally:
        torch.npu.config.allow_internal_format = old_allow
    assert torch_npu.get_npu_format(nz) in (FRACTAL_NZ, "FRACTAL_NZ")
    return nz


def _bf16_staged_reference(x, down, up, hc, hs):
    """PyTorch reference matching ascendc's BF16 projection boundaries."""
    rows = x.shape[0]
    if rows == 0:
        return x.new_empty((0, hs))
    low = F.linear(x, down)
    scaled = low.float() / hc
    activated = (scaled * (1.0 / (1.0 + torch.exp(-scaled)))).to(torch.bfloat16)
    gate_logits = F.linear(activated, up).reshape(rows, hc, hs)
    gate = 1.0 / (1.0 + torch.exp(-gate_logits.float()))
    mixed = (gate * x.reshape(rows, hc, hs).float()).sum(dim=1) / hc
    return mixed.to(torch.bfloat16)


def generate_data(shape, gate_mode, up_layout="ND", seed=11042026):
    """Create NPU inputs using the device context supplied by conftest."""
    rows, hc, hs, low_rank = shape
    total_width = hc * hs
    torch.npu.manual_seed(seed)
    x = torch.randn(rows, total_width, dtype=torch.bfloat16)
    down_scale = 2.0 * float(hc) * (total_width**-0.5)
    down = (torch.randn(low_rank, total_width, dtype=torch.float32) * down_scale).to(
        torch.bfloat16
    )
    if gate_mode == "normal":
        up_gain = 4.0
    elif gate_mode == "saturated":
        up_gain = 200.0
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    up = (
        torch.randn(total_width, low_rank, dtype=torch.float32)
        * (up_gain * (low_rank**-0.5))
    ).to(torch.bfloat16)
    if up_layout == "FRACTAL_NZ":
        up = _cast_fractal_nz(up)
    elif up_layout != "ND":
        raise ValueError(f"Unknown up_layout: {up_layout}")
    return x, down, up


@pytest.mark.parametrize("up_layout", UP_LAYOUTS)
@pytest.mark.parametrize("gate_mode", GATE_MODES)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 4, 2048, 320), id="qwen38-flash-next-decode-32"),
        pytest.param((64, 4, 2560, 320), id="qwen38-24l-decode-64"),
        pytest.param((2048, 4, 2048, 320), id="qwen38-flash-next-prefill-2048"),
        pytest.param((1, 4, 2048, 320), id="qwen38-flash-next-single-row"),
        pytest.param((16, 2, 2048, 128), id="two-branches"),
        pytest.param((1, 1, 16, 16), id="edge-minimum"),
        pytest.param((7, 2, 48, 32), id="edge-odd-rows-narrow-width"),
        pytest.param((3, 8, 4096, 64), id="edge-maximum-branches-and-width"),
        pytest.param((5, 4, 512, 4096), id="edge-maximum-lowrank"),
        pytest.param((0, 4, 2048, 320), id="edge-zero-rows"),
    ],
)
def test_hc_mix_ascendc(shape, gate_mode, up_layout):
    from sgl_kernel_npu.operators.hyper_conn import hc_mix

    require_distinct_impls(hc_mix, "ascendc", "torch_native")
    _, hc, hs, _ = shape
    x, down, up = generate_data(shape, gate_mode, up_layout)
    staged_ref = _bf16_staged_reference(x, down, up, hc, hs)
    if gate_mode == "normal":
        executor = check_accuracy(
            hc_mix,
            args=(x, down, up, hc, hs),
            impl="ascendc",
            impl_ref="torch_native",
            standard=BF16_VS_FP64_REF,
        )
        # Also enforce 1-2 BF16 ulp agreement on the AIV vector kernels themselves.
        assert_accuracy(executor(), staged_ref, BF16_TIGHT)
        check_performance(executor, enable_perf=shape in perf_shape)
    else:
        # With up_gain=200.0, BF16 vs FP64 projection rounding can flip logits
        # near zero; verify finite saturation and tight AIV agreement against
        # the BF16-projection staged reference.
        out = hc_mix(x, down, up, hc, hs, impl="ascendc")
        assert torch.isfinite(out).all()
        assert_accuracy(out, staged_ref, BF16_TIGHT)


def test_hc_mix_ascendc_graph_replay():
    """Captured Ascend C hc_mix graph must replay accurately after input mutations."""
    from sgl_kernel_npu.operators.hyper_conn import hc_mix

    require_distinct_impls(hc_mix, "ascendc", "torch_native")
    x, down, up = generate_data((16, 4, 2560, 320), "normal", "ND")
    for _ in range(3):
        hc_mix(x, down, up, 4, 2560, impl="ascendc")
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = hc_mix(x, down, up, 4, 2560, impl="ascendc")
    x.mul_(-0.5).add_(0.03125)
    graph.replay()
    torch.npu.synchronize()
    expected = _bf16_staged_reference(x, down, up, 4, 2560)
    assert_accuracy(out, expected, BF16_TIGHT)
    assert torch.equal(out, hc_mix(x, down, up, 4, 2560, impl="ascendc"))


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(1, id="rows-1"),
        pytest.param(8, id="rows-8"),
        pytest.param(31, id="rows-31"),
        pytest.param(64, id="rows-64"),
        pytest.param(129, id="rows-129-above-the-tuned-ladder"),
    ],
)
def test_hc_mix_fractal_nz_matches_row_major(rows):
    """A pre-converted FRACTAL_NZ up weight must agree with the ND up weight."""
    from sgl_kernel_npu.operators.hyper_conn import hc_mix

    require_test_impl(hc_mix, "ascendc")
    shape = (rows, 4, 2048, 320)
    x, down, up_row_major = generate_data(shape, "normal", "ND")
    up_nz = _cast_fractal_nz(up_row_major.clone())
    out_row_major = hc_mix(x, down, up_row_major, 4, 2048, impl="ascendc")
    out_nz = hc_mix(x, down, up_nz, 4, 2048, impl="ascendc")
    torch.testing.assert_close(out_nz, out_row_major, rtol=5e-2, atol=5e-2)


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
        pytest.param("normed_cols", "x must have shape", id="rejects-normed-cols"),
        pytest.param("down_cols", "down must have shape", id="rejects-down-cols"),
        pytest.param("low_rank", "low_rank", id="rejects-unaligned-lowrank"),
        pytest.param("up_shape", "up must have shape", id="rejects-up-shape"),
        pytest.param("strided", "contiguous", id="rejects-strided-input"),
        pytest.param("grad", "inference only", id="rejects-requires-grad"),
    ],
)
def test_hc_mix_rejects_inputs(case, message):
    """Unsupported inputs must raise ValueError before any implementation runs."""
    from sgl_kernel_npu.operators.hyper_conn import hc_mix

    x, down, up = generate_data((8, 4, 2048, 320), "normal")
    hc, hs = 4, 2048
    if case == "dtype":
        down = down.to(torch.float16)
    elif case == "device":
        x, down, up = x.cpu(), down.cpu(), up.cpu()
    elif case == "cross_device":
        up = up.cpu()
    elif case == "rank":
        x = x.reshape(8, 4, 2048)
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
    elif case == "normed_cols":
        x = torch.randn(8, 2048, dtype=torch.bfloat16)
    elif case == "down_cols":
        down = torch.randn(320, 2048, dtype=torch.bfloat16)
    elif case == "low_rank":
        down = torch.randn(312, 4 * 2048, dtype=torch.bfloat16)
        up = torch.randn(4 * 2048, 312, dtype=torch.bfloat16)
    elif case == "up_shape":
        up = torch.randn(2048, 320, dtype=torch.bfloat16)
    elif case == "strided":
        x = torch.randn(8, 2 * 4 * 2048, dtype=torch.bfloat16)[:, : 4 * 2048]
    elif case == "grad":
        x = x.requires_grad_()
    with pytest.raises(ValueError, match=message):
        hc_mix(x, down, up, hc, hs)
