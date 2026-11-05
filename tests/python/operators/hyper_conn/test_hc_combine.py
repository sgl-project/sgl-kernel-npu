"""Public hc_combine operator: injection projections, branch counts and guards.

Model dimensions follow Qwen3.8-Flash-Next (hidden_size=2048, hc_count=4;
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) and its
24-layer configuration (hidden_size=2560, hc_count=4), matching SGLang's
GatedResidual.combine in sglang/srt/layers/hyperconnection.py. Token counts are
representative decode batch buckets and prefill chunks.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# 1-2 BF16 ulp across a binade (1 ulp at 2^e is 2^-7 = 7.8125e-3), tight enough
# to catch accidental intermediate BF16 rounding (~0.01-0.02) while leaving room
# for fp32 transcendental and tree-reduction rounding differences.
BF16_TIGHT = {"torch.bfloat16": AccuracyStandard(atol=0.0, rtol=8e-3)}

# Production decode and prefill shapes selected for performance collection.
perf_shape = [
    (32, 4, 2048),
    (2048, 4, 2048),
]

GATE_MODES = [
    "normal",  # Projected injection logits spanning both sigmoid regions.
    "saturated",  # Large projection gain: gates saturate to 0 or 2.
]


def _native_available():
    """Report whether this build registered the A3-family native helper."""
    try:
        import sgl_kernel_npu  # noqa: F401

        return hasattr(torch.ops.npu, "hyper_conn_branch_combine")
    except Exception:
        return False


requires_native = pytest.mark.skipif(
    not _native_available(),
    reason="hc_combine needs an arch22 A3-family build of the native helper",
)


def generate_data(shape, gate_mode, seed=11042026):
    """Create NPU inputs using the device context supplied by conftest.

    shape is (rows, hc_count, hidden_size). inject_weight is scaled by fan-in
    so gate_mode controls the post-projection sigmoid regime.
    """
    rows, hc_count, hidden_size = shape
    total_width = hc_count * hidden_size
    torch.npu.manual_seed(seed)
    block_output = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
    residual = torch.randn(rows, total_width, dtype=torch.bfloat16)
    normed_residual = torch.randn(rows, total_width, dtype=torch.bfloat16)
    if gate_mode == "normal":
        gain = 4.0
    elif gate_mode == "saturated":
        gain = 200.0
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    scale = gain * float(hc_count) * (total_width**-0.5)
    inject_weight = (
        torch.randn(hc_count, total_width, dtype=torch.float32) * scale
    ).to(torch.bfloat16)
    return block_output, residual, normed_residual, inject_weight


@requires_native
@pytest.mark.parametrize("gate_mode", GATE_MODES)
@pytest.mark.parametrize(
    "shape",
    [
        # Qwen3.8-Flash-Next: hidden_size=2048, hc_count=4; 24-layer variant:
        # hidden_size=2560, hc_count=4.
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
def test_hc_combine(shape, gate_mode):
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    _, hc_count, hidden_size = shape
    inputs = generate_data(shape, gate_mode)
    executor = check_accuracy(
        hc_combine,
        args=(*inputs, hc_count, hidden_size),
        standard=BF16_TIGHT,
    )
    # An accuracy failure raises before performance collection.
    check_performance(
        executor, enable_perf=shape in perf_shape and gate_mode == "normal"
    )


@pytest.mark.extra_checks
@requires_native
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 4, 2048), id="decode-32x4x2048"),
        pytest.param((1, 1, 16), id="edge-minimum"),
    ],
)
def test_hc_combine_leaves_inputs_unchanged(shape, impl_pair):
    """The residual is returned as a new tensor, not updated in place."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    impl = impl_pair[0]
    _, hc_count, hidden_size = shape
    block_output, residual, normed_residual, inject_weight = generate_data(
        shape, "normal"
    )
    before = residual.clone()
    result = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        hc_count,
        hidden_size,
        impl=impl,
    )
    assert result.data_ptr() != residual.data_ptr()
    assert torch.equal(residual, before)


@pytest.mark.extra_checks
@requires_native
@pytest.mark.parametrize("impl", ["ascendc", "torch_native"])
def test_hc_combine_out_buffer(impl):
    """Pre-allocated out buffers are written in place without mutating inputs."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    shape = (16, 4, 2048)
    _, hc_count, hidden_size = shape
    block_output, residual, normed_residual, inject_weight = generate_data(
        shape, "normal"
    )
    baseline = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        hc_count,
        hidden_size,
        impl=impl,
    )

    out_kw = torch.full_like(residual, -1.0)
    ptr_kw = out_kw.data_ptr()
    ret_kw = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        hc_count,
        hidden_size,
        out=out_kw,
        impl=impl,
    )
    assert ret_kw is out_kw
    assert ret_kw.data_ptr() == ptr_kw
    assert torch.equal(out_kw, baseline)

    out_pos = torch.full_like(residual, -1.0)
    ptr_pos = out_pos.data_ptr()
    ret_pos = hc_combine(
        block_output,
        residual,
        normed_residual,
        inject_weight,
        hc_count,
        hidden_size,
        out_pos,
        impl=impl,
    )
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
        block_output,
        res_slice,
        normed_residual,
        inject_weight,
        hc_count,
        hidden_size,
        out=out_slice,
        impl=impl,
    )
    assert ret_slice is out_slice
    assert ret_slice.data_ptr() == ptr_slice
    assert torch.equal(out_slice, baseline)
    assert torch.equal(res_slice, residual)

    block0, res0, normed0, weight0 = generate_data((0, hc_count, hidden_size), "normal")
    out0 = torch.empty(
        (0, hc_count * hidden_size), dtype=torch.bfloat16, device=residual.device
    )
    ret0 = hc_combine(
        block0, res0, normed0, weight0, hc_count, hidden_size, out=out0, impl=impl
    )
    assert ret0 is out0
    assert ret0.shape == (0, hc_count * hidden_size)


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

    block_output, residual, normed_residual, inject_weight = generate_data(
        (8, 4, 2048), "normal"
    )
    hc_count, hidden_size = 4, 2048
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
        out = normed_residual
    elif case == "partial_overlap":
        n = residual.numel()
        pool = torch.empty(n + n // 2, dtype=torch.bfloat16, device=residual.device)
        residual = pool[:n].view_as(residual)
        out = pool[n // 2 : n + n // 2].view_as(residual)
    elif case == "zero_row_alias":
        block_output, residual, normed_residual, inject_weight = generate_data(
            (0, 4, 2048), "normal"
        )
        out = residual
    with pytest.raises(ValueError, match=message):
        hc_combine(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            hc_count,
            hidden_size,
            out=out,
        )


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
        pytest.param(
            "residual_shape", "residual must have shape", id="rejects-residual"
        ),
        pytest.param(
            "normed_shape", "normed_residual must have shape", id="rejects-normed"
        ),
        pytest.param("block_shape", "block_output must have shape", id="rejects-block"),
        pytest.param(
            "inject_shape", "inject_weight must have shape", id="rejects-inject"
        ),
        pytest.param("strided", "contiguous", id="rejects-strided-residual"),
        pytest.param("grad", "inference only", id="rejects-requires-grad"),
    ],
)
def test_hc_combine_rejects_inputs(case, message):
    """Unsupported inputs must raise ValueError before any implementation runs."""
    from sgl_kernel_npu.operators.hyper_conn import hc_combine

    block_output, residual, normed_residual, inject_weight = generate_data(
        (8, 4, 2048), "normal"
    )
    hc_count, hidden_size = 4, 2048
    if case == "dtype":
        block_output = block_output.to(torch.float16)
    elif case == "device":
        block_output, residual, normed_residual, inject_weight = (
            block_output.cpu(),
            residual.cpu(),
            normed_residual.cpu(),
            inject_weight.cpu(),
        )
    elif case == "cross_device":
        inject_weight = inject_weight.cpu()
    elif case == "rank":
        residual = residual.reshape(8, 4, 2048)
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
    elif case == "residual_shape":
        residual = torch.randn(8, 2048, dtype=torch.bfloat16)
    elif case == "normed_shape":
        normed_residual = torch.randn(8, 2048, dtype=torch.bfloat16)
    elif case == "block_shape":
        block_output = torch.randn(8, 1024, dtype=torch.bfloat16)
    elif case == "inject_shape":
        inject_weight = torch.randn(3, 4 * 2048, dtype=torch.bfloat16)
    elif case == "strided":
        residual = torch.randn(8, 2 * 4 * 2048, dtype=torch.bfloat16)[:, : 4 * 2048]
    elif case == "grad":
        residual = residual.requires_grad_()
    with pytest.raises(ValueError, match=message):
        hc_combine(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            hc_count,
            hidden_size,
        )
