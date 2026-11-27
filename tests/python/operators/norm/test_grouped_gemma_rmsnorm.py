"""Public grouped_gemma_rmsnorm operator: Triton default and explicit Ascend C backend."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import require_distinct_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Allow BF16 output rounding around FP32/FP64 computation differences.
BF16_TIGHT = {"torch.bfloat16": AccuracyStandard(atol=0.0, rtol=8e-3)}
STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.005, rtol=0.01)}

# Production decode and prefill shapes selected for performance collection.
perf_shape = [
    (32, 4, 2048),
    (2048, 4, 2048),
]
perf_weight = "per_branch"
perf_2560_rows = [1, 32, 4096]

# How long the weight is relative to one group.
WEIGHT_LAYOUTS = [
    "per_branch",  # One slice per branch: the offset cycles over the row.
    "shared",  # One group long: every group uses the same slice.
]


def _supported_generation():
    return get_hardware().generation in {Generation.ARCH22, Generation.ARCH35}


def generate_data(shape, weight_layout, seed=11042026):
    """Create NPU inputs using the device context supplied by conftest."""
    if weight_layout not in WEIGHT_LAYOUTS:
        raise ValueError(f"Unknown weight layout: {weight_layout}")
    rows, branches, width = shape
    torch.npu.manual_seed(seed)
    x = torch.randn(rows, branches * width, dtype=torch.bfloat16)
    length = width if weight_layout == "shared" else branches * width
    weight = (torch.randn(length, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    return x, weight


def _generate_2560_data(rows, row_offset=0):
    generator = torch.Generator(device="cpu").manual_seed(73)
    x = torch.randn(rows + row_offset, 10240, generator=generator, device="cpu").to(
        torch.bfloat16
    )
    weight = (torch.randn(10240, generator=generator, device="cpu") * 0.1).to(
        torch.bfloat16
    )
    # Slice after transfer: SP rank 1 keeps a nonzero offset on the device.
    x = x.to("npu")[row_offset:]
    assert x.is_contiguous() and x.storage_offset() == row_offset * 10240
    return x, weight.to("npu"), 2560, 1e-6


@pytest.mark.parametrize(
    "rows,row_offset",
    [
        pytest.param(1, 0, id="qwen3-8-flash-next-decode-b1-h10240"),
        pytest.param(8, 0, id="qwen3-8-flash-next-decode-b8-h10240"),
        pytest.param(4, 0, id="qwen3-8-flash-next-verify-b1-w4-t4-h10240"),
        pytest.param(32, 0, id="qwen3-8-flash-next-verify-b8-w4-t32-h10240"),
        pytest.param(4096, 0, id="qwen3-8-flash-next-prefill-t4096-h10240"),
        pytest.param(
            512, 512, id="qwen3-8-flash-next-sp8-prefill-local-t512-offset512"
        ),
        pytest.param(0, 0, id="edge-empty"),
        pytest.param(33, 0, id="edge-row-tail-r33"),
        pytest.param(8193, 0, id="edge-large-row-tail-r8193"),
    ],
)
def test_grouped_gemma_rmsnorm_accuracy(rows, row_offset):
    if not _supported_generation():
        pytest.skip("These model operators target arch22 and arch35")
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    args = _generate_2560_data(rows, row_offset)
    saved = [x.clone() for x in args[:2]]
    executor = check_accuracy(grouped_gemma_rmsnorm, args=args, standard=STANDARD)
    assert executor().is_contiguous()
    _assert_unchanged(args, saved)
    check_performance(executor, enable_perf=rows in perf_2560_rows)


@pytest.mark.parametrize(
    "rows", [pytest.param(8, id="qwen3-8-flash-next-decode-r8-h10240")]
)
def test_grouped_gemma_rmsnorm_graph_updates(rows):
    if not _supported_generation():
        pytest.skip("These model operators target arch22 and arch35")
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    args = _generate_2560_data(rows)
    _check_graph_updates(
        grouped_gemma_rmsnorm,
        args,
        [lambda: args[0].mul_(-0.5), lambda: args[1].add_(0.125)],
        standard=STANDARD,
    )


@pytest.mark.parametrize("weight_layout", WEIGHT_LAYOUTS)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 4, 2048), id="qwen38-flash-next-decode-32x4x2048"),
        pytest.param((64, 4, 2560), id="qwen38-24l-decode-64x4x2560"),
        pytest.param((2048, 4, 2048), id="qwen38-flash-next-prefill-2048x4x2048"),
        pytest.param((1, 4, 2048), id="qwen38-flash-next-single-row"),
        pytest.param((1, 1, 8), id="edge-minimum"),
        pytest.param((7, 2, 24), id="edge-odd-rows-narrow-group"),
        pytest.param((5, 8, 4096), id="edge-maximum-width"),
        pytest.param((0, 4, 2048), id="edge-zero-rows"),
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_grouped_gemma_rmsnorm_ascendc(shape, weight_layout, eps):
    """Explicit Ascend C backend over general widths, layouts and eps values."""
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    require_distinct_impls(grouped_gemma_rmsnorm, "ascendc", "torch_native")
    x, weight = generate_data(shape, weight_layout)
    executor = check_accuracy(
        grouped_gemma_rmsnorm,
        args=(x, weight, shape[2], eps),
        impl="ascendc",
        impl_ref="torch_native",
        standard=BF16_TIGHT,
    )
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape and weight_layout == perf_weight and eps == 1e-6
        ),
    )


@pytest.mark.parametrize(
    "rows", [pytest.param(8, id="qwen3-8-flash-next-decode-r8-h10240")]
)
def test_grouped_gemma_rmsnorm_ascendc_graph_updates(rows):
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    require_distinct_impls(grouped_gemma_rmsnorm, "ascendc", "torch_native")
    args = _generate_2560_data(rows)
    _check_graph_updates(
        grouped_gemma_rmsnorm,
        args,
        [lambda: args[0].mul_(-0.5), lambda: args[1].add_(0.125)],
        standard=STANDARD,
        kwargs={"impl": "ascendc"},
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((16, 2, 1024), id="decode-16x2048"),
        pytest.param((2048, 2, 1280), id="prefill-2048x2560"),
        pytest.param((1, 1, 16), id="edge-minimum"),
    ],
)
def test_grouped_gemma_rmsnorm_ascendc_defaults(shape):
    """group_size=None and omitted eps must match explicit (row_width, 1e-6)."""
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    require_distinct_impls(grouped_gemma_rmsnorm, "ascendc", "torch_native")
    rows, branches, width = shape
    row_width = branches * width
    x, _ = generate_data(shape, "per_branch")
    weight = (torch.randn(row_width, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    executor = check_accuracy(
        grouped_gemma_rmsnorm,
        args=(x, weight, None),
        impl="ascendc",
        impl_ref="torch_native",
        standard=BF16_TIGHT,
    )
    explicit = grouped_gemma_rmsnorm(x, weight, row_width, 1e-6, impl="ascendc")
    assert torch.equal(executor(), explicit)
    assert torch.equal(grouped_gemma_rmsnorm(x, weight, impl="ascendc"), explicit)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((8, 4, 2048), id="qwen38-flash-next-3d-per-branch"),
        pytest.param((8, 4, 2048), id="qwen38-flash-next-3d-shared"),
        pytest.param((3, 2, 128), id="edge-small-3d"),
    ],
)
def test_grouped_gemma_rmsnorm_ascendc_three_dimensional(shape, request):
    """A [rows, branches, width] input must group identically to the flat view."""
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    require_distinct_impls(grouped_gemma_rmsnorm, "ascendc", "torch_native")
    rows, branches, width = shape
    layout = "shared" if "shared" in request.node.callspec.id else "per_branch"
    x, weight = generate_data(shape, layout)
    flat = grouped_gemma_rmsnorm(x, weight, width, 1e-6, impl="ascendc")
    view_3d = x.reshape(rows, branches, width)
    group_arg = None if layout == "shared" else width
    executor = check_accuracy(
        grouped_gemma_rmsnorm,
        args=(view_3d, weight, group_arg, 1e-6),
        impl="ascendc",
        impl_ref="torch_native",
        standard=BF16_TIGHT,
    )
    assert torch.equal(executor().reshape(rows, branches * width), flat)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["ascendc", "torch_native"])
def test_grouped_gemma_rmsnorm_out_buffer(impl):
    """Pre-allocated out buffers are written in place without mutating inputs."""
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    require_test_impl(grouped_gemma_rmsnorm, impl)
    shape = (16, 4, 2048)
    _, branches, width = shape
    x, weight = generate_data(shape, "per_branch")
    baseline = grouped_gemma_rmsnorm(x, weight, width, 1e-6, impl=impl)

    out_kw = torch.full_like(x, -1.0)
    ptr_kw = out_kw.data_ptr()
    ret_kw = grouped_gemma_rmsnorm(x, weight, width, 1e-6, out=out_kw, impl=impl)
    assert ret_kw is out_kw
    assert ret_kw.data_ptr() == ptr_kw
    assert torch.equal(out_kw, baseline)

    out_pos = torch.full_like(x, -1.0)
    ptr_pos = out_pos.data_ptr()
    ret_pos = grouped_gemma_rmsnorm(x, weight, width, 1e-6, out_pos, impl=impl)
    assert ret_pos is out_pos
    assert ret_pos.data_ptr() == ptr_pos
    assert torch.equal(out_pos, baseline)

    n = x.numel()
    pool = torch.empty(2 * n, dtype=torch.bfloat16, device=x.device)
    x_slice = pool[:n].view_as(x)
    x_slice.copy_(x)
    out_slice = pool[n : 2 * n].view_as(x)
    out_slice.fill_(-1.0)
    ptr_slice = out_slice.data_ptr()
    ret_slice = grouped_gemma_rmsnorm(
        x_slice, weight, width, 1e-6, out=out_slice, impl=impl
    )
    assert ret_slice is out_slice
    assert ret_slice.data_ptr() == ptr_slice
    assert torch.equal(out_slice, baseline)
    assert torch.equal(x_slice, x)

    x0 = torch.empty((0, branches * width), dtype=torch.bfloat16, device=x.device)
    out0 = torch.empty((0, branches * width), dtype=torch.bfloat16, device=x.device)
    ret0 = grouped_gemma_rmsnorm(x0, weight, width, 1e-6, out=out0, impl=impl)
    assert ret0 is out0
    assert ret0.shape == (0, branches * width)


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
        pytest.param("alias_x", "alias or overlap", id="rejects-out-aliasing-x"),
        pytest.param(
            "alias_weight", "alias or overlap", id="rejects-out-overlapping-weight"
        ),
        pytest.param(
            "partial_overlap", "alias or overlap", id="rejects-partial-overlap-out"
        ),
        pytest.param(
            "zero_row_alias", "alias or overlap", id="rejects-zero-row-self-alias"
        ),
    ],
)
def test_grouped_gemma_rmsnorm_rejects_invalid_out(case, message):
    """Invalid out buffers must raise ValueError before any kernel launch."""
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    x, weight = generate_data((8, 4, 2048), "per_branch")
    group_size, eps = 2048, 1e-6
    if case == "non_tensor":
        out = "bad"
    elif case == "dtype":
        out = torch.empty_like(x, dtype=torch.float32)
    elif case == "cross_device":
        out = torch.empty_like(x, device="cpu")
    elif case == "shape":
        out = torch.empty((8, 2048), dtype=torch.bfloat16, device=x.device)
    elif case == "strided":
        out = torch.empty((8, 2 * 4 * 2048), dtype=torch.bfloat16, device=x.device)[
            :, : 4 * 2048
        ]
    elif case == "grad":
        out = torch.empty_like(x).requires_grad_(True)
    elif case == "alias_x":
        out = x
    elif case == "alias_weight":
        x = x[:1]
        out = weight.view_as(x)
    elif case == "partial_overlap":
        n = x.numel()
        pool = torch.empty(n + n // 2, dtype=torch.bfloat16, device=x.device)
        x = pool[:n].view_as(x)
        out = pool[n // 2 : n + n // 2].view_as(x)
    elif case == "zero_row_alias":
        x = torch.empty((0, 4 * 2048), dtype=torch.bfloat16, device=x.device)
        out = x
    with pytest.raises(ValueError, match=message):
        grouped_gemma_rmsnorm(x, weight, group_size, eps, out=out)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case,message",
    [
        pytest.param("dtype", "BF16", id="rejects-float16"),
        pytest.param("device", "same NPU", id="rejects-cpu"),
        pytest.param("cross_device", "same NPU", id="rejects-cross-device"),
        pytest.param("rank", "at least 2 dimensions", id="rejects-1d"),
        pytest.param("bool_group", "integer or None", id="rejects-bool-group-size"),
        pytest.param("group_size", "group size", id="rejects-unaligned-group-size"),
        pytest.param("too_wide", "group size", id="rejects-group-size-above-4096"),
        pytest.param("indivisible", "group size", id="rejects-indivisible-row"),
        pytest.param("zero_cols", "group size", id="rejects-zero-last-dim"),
        pytest.param(
            "weight_length", "multiple of the group size", id="rejects-weight-length"
        ),
        pytest.param(
            "weight_period", "weight periods", id="rejects-partial-weight-period"
        ),
        pytest.param("bool_eps", "real number", id="rejects-bool-eps"),
        pytest.param("eps", "eps", id="rejects-nonpositive-eps"),
        pytest.param("nan_eps", "positive and finite", id="rejects-nan-eps"),
        pytest.param("inf_eps", "positive and finite", id="rejects-inf-eps"),
        pytest.param("strided", "contiguous", id="rejects-strided-input"),
        pytest.param("grad", "inference only", id="rejects-requires-grad"),
    ],
)
def test_grouped_gemma_rmsnorm_rejects_inputs(case, message):
    """Unsupported inputs must raise ValueError before any implementation runs."""
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    x, weight = generate_data((8, 4, 2048), "per_branch")
    group_size, eps = 2048, 1e-6
    if case == "dtype":
        x = x.to(torch.float16)
    elif case == "device":
        x, weight = x.cpu(), weight.cpu()
    elif case == "cross_device":
        weight = weight.cpu()
    elif case == "rank":
        x = x.reshape(-1)
    elif case == "bool_group":
        group_size = True
    elif case == "group_size":
        group_size = 2044
    elif case == "too_wide":
        x = torch.randn(8, 8192, dtype=torch.bfloat16)
        weight = torch.randn(8192, dtype=torch.bfloat16)
        group_size = 8192
    elif case == "indivisible":
        group_size = 1032
    elif case == "zero_cols":
        x = torch.empty((8, 0), dtype=torch.bfloat16)
        weight = torch.randn(2048, dtype=torch.bfloat16)
    elif case == "weight_length":
        weight = torch.randn(2048 + 8, dtype=torch.bfloat16)
    elif case == "weight_period":
        weight = torch.randn(3 * 2048, dtype=torch.bfloat16)
    elif case == "bool_eps":
        eps = True
    elif case == "eps":
        eps = 0.0
    elif case == "nan_eps":
        eps = float("nan")
    elif case == "inf_eps":
        eps = float("inf")
    elif case == "strided":
        x = torch.randn(8, 2 * 4 * 2048, dtype=torch.bfloat16)[:, : 4 * 2048]
    elif case == "grad":
        x = x.requires_grad_()
    with pytest.raises(ValueError, match=message):
        grouped_gemma_rmsnorm(x, weight, group_size, eps)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "eps",
    [0, -1, float("nan"), float("inf"), True],
    ids=["zero", "negative", "nan", "infinite", "boolean"],
)
def test_grouped_gemma_rmsnorm_rejects_epsilon(eps):
    from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

    x, weight, group, _ = _generate_2560_data(0)
    with pytest.raises(ValueError):
        grouped_gemma_rmsnorm(x, weight, group, eps)


def _tensors(args):
    return [value for value in args if isinstance(value, torch.Tensor)]


def _assert_unchanged(args, saved):
    for value, before in zip(_tensors(args), saved):
        torch.testing.assert_close(value, before, atol=0, rtol=0, equal_nan=True)


def _check_graph_updates(
    operator, args, updates, *, standard=None, kwargs=None, compare=None
):
    """Capture a call; compare each replay to an independent reference."""
    kwargs = dict(kwargs or {})
    inputs = _tensors(args)
    pointers = [x.data_ptr() for x in inputs]
    for _ in range(3):
        operator(*args, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operator(*args, **kwargs)
    for change in [None, *updates]:
        if change is not None:
            change()
        saved = [x.clone() for x in inputs]
        graph.replay()
        torch.npu.synchronize()
        if compare is None:
            ref_kwargs = {k: v for k, v in kwargs.items() if k != "impl"}
            reference = operator(*args, **ref_kwargs, impl="torch_native")
            assert_accuracy(output, reference, standard)
        else:
            compare(output)
        assert [x.data_ptr() for x in inputs] == pointers
        _assert_unchanged(args, saved)
    return graph, output
