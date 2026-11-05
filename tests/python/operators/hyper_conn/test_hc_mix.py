"""Public hc_mix operator: low-rank projections, NZ weights and restrictions.

Model dimensions follow Qwen3.8-Flash-Next (hidden_size=2048, hc_count=4,
hc_lowrank=320; https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
and its 24-layer configuration (hidden_size=2560, hc_count=4, hc_lowrank=320),
matching SGLang's GatedResidual.mix in sglang/srt/layers/hyperconnection.py.
Token counts are representative decode batch buckets and prefill chunks.
"""

import importlib.util

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# 1-2 BF16 ulp across a binade (1 ulp at 2^e is 2^-7 = 7.8125e-3), tight enough
# to catch accidental intermediate BF16 rounding (~0.01-0.02) while leaving room
# for fp32 transcendental and tree-reduction rounding differences.
BF16_TIGHT = {"torch.bfloat16": AccuracyStandard(atol=0.0, rtol=8e-3)}

FRACTAL_NZ = 29

# Production decode and prefill shapes (rows, hc_count, hidden_size, low_rank)
# selected for performance collection across both ND and FRACTAL_NZ up weights.
perf_shape = [
    (32, 4, 2048, 320),
    (2048, 4, 2048, 320),
]

GATE_MODES = [
    "normal",  # Gate logits spanning the curved sigmoid region and moderate tails.
    "saturated",  # Large projection gain: gates saturate cleanly to 0 or 1.
]

UP_LAYOUTS = [
    "ND",  # Standard row-major [hc_count * hidden_size, low_rank] weight.
    "FRACTAL_NZ",  # Pre-converted FRACTAL_NZ weight for faster decode GEMM.
]


def _native_available():
    """Report whether this build registered the A3-family native helpers."""
    try:
        import sgl_kernel_npu  # noqa: F401

        return hasattr(torch.ops.npu, "hyper_conn_scaled_silu") and hasattr(
            torch.ops.npu, "hyper_conn_branch_mix"
        )
    except Exception:
        return False


requires_native = pytest.mark.skipif(
    not _native_available(),
    reason="hc_mix needs an arch22 A3-family build of the native helpers",
)


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


def generate_data(shape, gate_mode, up_layout="ND", seed=11042026):
    """Create NPU inputs using the device context supplied by conftest.

    shape is (rows, hc_count, hidden_size, low_rank). Projection weights are
    scaled by fan-in so the down projection hits the curved region of scaled
    SiLU and gate_mode controls the up-projected sigmoid regime.
    """
    rows, hc_count, hidden_size, low_rank = shape
    total_width = hc_count * hidden_size
    torch.npu.manual_seed(seed)
    normed = torch.randn(rows, total_width, dtype=torch.bfloat16)
    down_scale = 2.0 * float(hc_count) * (total_width**-0.5)
    down_weight = (
        torch.randn(low_rank, total_width, dtype=torch.float32) * down_scale
    ).to(torch.bfloat16)
    if gate_mode == "normal":
        up_gain = 4.0
    elif gate_mode == "saturated":
        up_gain = 200.0
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    up_weight = (
        torch.randn(total_width, low_rank, dtype=torch.float32)
        * (up_gain * (low_rank**-0.5))
    ).to(torch.bfloat16)
    if up_layout == "FRACTAL_NZ":
        up_weight = _cast_fractal_nz(up_weight)
    elif up_layout != "ND":
        raise ValueError(f"Unknown up_layout: {up_layout}")
    return normed, down_weight, up_weight


@requires_native
@pytest.mark.parametrize("up_layout", UP_LAYOUTS)
@pytest.mark.parametrize("gate_mode", GATE_MODES)
@pytest.mark.parametrize(
    "shape",
    [
        # Qwen3.8-Flash-Next: hidden_size=2048, hc_count=4, hc_lowrank=320;
        # 24-layer variant: hidden_size=2560, hc_count=4, hc_lowrank=320.
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
def test_hc_mix(shape, gate_mode, up_layout):
    from sgl_kernel_npu.operators.hyper_conn import hc_mix

    _, hc_count, hidden_size, _ = shape
    normed, down_weight, up_weight = generate_data(shape, gate_mode, up_layout)
    executor = check_accuracy(
        hc_mix,
        args=(normed, down_weight, up_weight, hc_count, hidden_size),
        standard=BF16_TIGHT,
    )
    # An accuracy failure raises before performance collection.
    check_performance(
        executor,
        enable_perf=shape in perf_shape and gate_mode == "normal",
    )


@pytest.mark.extra_checks
@requires_native
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
def test_hc_mix_fractal_nz_matches_row_major(rows, impl_pair):
    """A pre-converted FRACTAL_NZ up_weight must agree with the ND up_weight."""
    from sgl_kernel_npu.operators.hyper_conn import hc_mix

    impl = impl_pair[0]
    shape = (rows, 4, 2048, 320)
    normed, down_weight, up_row_major = generate_data(shape, "normal", "ND")
    up_nz = _cast_fractal_nz(up_row_major.clone())
    out_row_major = hc_mix(normed, down_weight, up_row_major, 4, 2048, impl=impl)
    out_nz = hc_mix(normed, down_weight, up_nz, 4, 2048, impl=impl)
    torch.testing.assert_close(out_nz, out_row_major, rtol=5e-2, atol=5e-2)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,message",
    [
        pytest.param("dtype", "BF16", id="rejects-float16"),
        pytest.param("device", "same NPU", id="rejects-cpu"),
        pytest.param("cross_device", "same NPU", id="rejects-cross-device"),
        pytest.param("rank", "2-D", id="rejects-3d"),
        pytest.param("bool_hc", "hc_count", id="rejects-bool-hc-count"),
        pytest.param("hc_count", "power of two", id="rejects-three-branches"),
        pytest.param("too_many", "power of two", id="rejects-sixteen-branches"),
        pytest.param("bool_hidden", "hidden_size", id="rejects-bool-hidden-size"),
        pytest.param("hidden_size", "multiple of 16", id="rejects-unaligned-hidden"),
        pytest.param("normed_cols", "normed must have shape", id="rejects-normed-cols"),
        pytest.param(
            "down_cols", "down_weight must have shape", id="rejects-down-cols"
        ),
        pytest.param("low_rank", "low_rank", id="rejects-unaligned-lowrank"),
        pytest.param("up_shape", "up_weight must have shape", id="rejects-up-shape"),
        pytest.param("strided", "contiguous", id="rejects-strided-input"),
        pytest.param("grad", "inference only", id="rejects-requires-grad"),
    ],
)
def test_hc_mix_rejects_inputs(case, message):
    """Unsupported inputs must raise ValueError before any implementation runs."""
    from sgl_kernel_npu.operators.hyper_conn import hc_mix

    normed, down_weight, up_weight = generate_data((8, 4, 2048, 320), "normal")
    hc_count, hidden_size = 4, 2048
    if case == "dtype":
        down_weight = down_weight.to(torch.float16)
    elif case == "device":
        normed, down_weight, up_weight = (
            normed.cpu(),
            down_weight.cpu(),
            up_weight.cpu(),
        )
    elif case == "cross_device":
        up_weight = up_weight.cpu()
    elif case == "rank":
        normed = normed.reshape(8, 4, 2048)
    elif case == "bool_hc":
        hc_count = True
    elif case == "hc_count":
        hc_count = 3
    elif case == "too_many":
        hc_count = 16
    elif case == "bool_hidden":
        hidden_size = True
    elif case == "hidden_size":
        hidden_size = 24
    elif case == "normed_cols":
        normed = torch.randn(8, 2048, dtype=torch.bfloat16)
    elif case == "down_cols":
        down_weight = torch.randn(320, 2048, dtype=torch.bfloat16)
    elif case == "low_rank":
        down_weight = torch.randn(312, 4 * 2048, dtype=torch.bfloat16)
        up_weight = torch.randn(4 * 2048, 312, dtype=torch.bfloat16)
    elif case == "up_shape":
        up_weight = torch.randn(2048, 320, dtype=torch.bfloat16)
    elif case == "strided":
        normed = torch.randn(8, 2 * 4 * 2048, dtype=torch.bfloat16)[:, : 4 * 2048]
    elif case == "grad":
        normed = normed.requires_grad_()
    with pytest.raises(ValueError, match=message):
        hc_mix(normed, down_weight, up_weight, hc_count, hidden_size)
