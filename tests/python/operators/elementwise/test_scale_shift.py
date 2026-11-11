"""Required SGLang scenarios first; supplementary checks are optional.

SGLang's multimodal_gen/runtime/layers/elementwise.py::MulAdd.forward_npu
passes [B, L, H] tensors with scale_constant=k. The layernorm.py NPU path
also passes scalar/channel modulation. wanvideo.py builds modulation from
temb.float(), so required cases include FP16/BF16 x with FP32 scale/shift.
Wan's H = 40 heads * 128 = 5120; token counts 128/257 are representative
video-token tiles, not runtime traces. The legacy [3, 37440, 5120] case's
batch and width are covered by a smaller optional allocation.

Default tests cover caller scenarios and three known BF16 rounding paths.
Generalization, dtype permutations and invalid inputs use extra_checks.
"""

from itertools import product

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(1, 128, 5120), (1, 257, 5120)]

modulation_cases = [
    pytest.param("scalar", "scalar", id="scalar-scale-scalar-shift"),
    pytest.param("scalar", "channel", id="scalar-scale-channel-shift"),
    pytest.param("channel", "scalar", id="channel-scale-scalar-shift"),
    pytest.param("channel", "channel", id="channel-scale-channel-shift"),
    pytest.param("channel", "elementwise", id="channel-scale-elementwise-shift"),
]


def generate_test_data(shape, dtype, scale_mode, shift_mode, seed=42):
    """Generate finite NPU inputs with modest magnitudes."""
    torch.npu.manual_seed(seed)
    hidden_size = shape[-1]
    scale_shape = (1,) if scale_mode == "scalar" else (1, 1, hidden_size)
    shift_shape = {
        "scalar": (1,),
        "channel": (1, 1, hidden_size),
        "elementwise": shape,
    }[shift_mode]
    return (
        torch.randn(shape, dtype=dtype) * 0.25,
        torch.randn(scale_shape, dtype=dtype) * 0.25,
        torch.randn(shift_shape, dtype=dtype) * 0.25,
    )


def _check_bf16_scale_rounding(shape, x_dtype, scale_mode, shift_mode, expected_factor):
    """Use exact values to detect missing or extra BF16 factor rounding."""
    if get_skip_acc_check() is True:
        pytest.skip("Output checks are disabled by --skip-acc-check=true")

    from sgl_kernel_npu.operators.elementwise import fused_scale_shift

    # Dyadic values keep multiply/add exact, even for float16 outputs.
    numel = shape[0] * shape[1] * shape[2]
    x = ((torch.arange(numel) % 4) - 2).reshape(shape).to(x_dtype)
    scale_shape = (1,) if scale_mode == "scalar" else (1, 1, shape[-1])
    shift_shape = {
        "scalar": (1,),
        "channel": (1, 1, shape[-1]),
        "elementwise": shape,
    }[shift_mode]
    shift = torch.full(shift_shape, 0.25, dtype=torch.float32)
    scale_value = 1 / 1024 if shift.numel() == x.numel() else 1 / 256
    scale = torch.full(scale_shape, scale_value, dtype=torch.bfloat16)
    expected = (x.float() * expected_factor + 0.25).to(x_dtype)
    executor = check_accuracy(
        fused_scale_shift,
        args=(x, scale, shift),
        # Also verify that broadcast shift ignores this negative constant.
        kwargs={"scale_constant": -0.5},
        standard={str(x_dtype): AccuracyStandard(atol=0, rtol=0)},
    )
    torch.testing.assert_close(executor(), expected, atol=0, rtol=0)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "dtypes",
    [
        pytest.param((torch.float32,) * 3, id="fp32"),
        pytest.param((torch.float16,) * 3, id="fp16"),
        pytest.param((torch.bfloat16,) * 3, id="bf16"),
        pytest.param(
            (torch.float16, torch.float32, torch.float32), id="fp16-fp32-modulation"
        ),
        pytest.param(
            (torch.bfloat16, torch.float32, torch.float32), id="bf16-fp32-modulation"
        ),
    ],
)
@pytest.mark.parametrize(
    "shape", perf_shape, ids=["wan-width-token-tile", "wan-width-token-tail"]
)
@pytest.mark.parametrize(
    "scale_mode,shift_mode,scale_constant",
    [
        pytest.param("scalar", "scalar", 1.0, id="scalar-modulation"),
        pytest.param("channel", "channel", 1.0, id="channel-modulation"),
        pytest.param("channel", "elementwise", 1.0, id="elementwise-shift"),
        pytest.param("channel", "elementwise", 0.0, id="muladd-zero-constant"),
    ],
)
def test_fused_scale_shift(shape, dtypes, scale_mode, shift_mode, scale_constant):
    """Cover representative SGLang inputs and both kernel paths."""
    from sgl_kernel_npu.operators.elementwise import fused_scale_shift

    inputs = generate_test_data(shape, torch.float32, scale_mode, shift_mode)
    inputs = tuple(value.to(dtype) for value, dtype in zip(inputs, dtypes))
    executor = check_accuracy(
        fused_scale_shift, args=inputs, kwargs={"scale_constant": scale_constant}
    )
    check_performance(
        executor,
        # Benchmark both kernel paths on each representative shape.
        enable_perf=(
            dtypes == (torch.float32,) * 3
            and scale_mode == "channel"
            and scale_constant == 1.0
        ),
    )


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "scale_mode,shift_mode,expected_factor",
    [
        pytest.param("scalar", "scalar", 1.0, id="scalar-load"),
        pytest.param("channel", "channel", 1.00390625, id="vector-load"),
        pytest.param("channel", "elementwise", -0.4990234375, id="full-shift"),
    ],
)
def test_fused_scale_shift_bf16_scale_rounding(
    x_dtype, scale_mode, shift_mode, expected_factor
):
    """Retain one regression per observed BF16 factor rounding path."""
    _check_bf16_scale_rounding(
        (2, 3, 129), x_dtype, scale_mode, shift_mode, expected_factor
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 5, 5120), (2, 3, 1), (1, 7, 129)])
@pytest.mark.parametrize(
    "scale_mode,shift_mode,scale_constant",
    [
        ("scalar", "scalar", 1.0),
        ("scalar", "channel", 1.0),
        ("channel", "scalar", 1.0),
        ("channel", "channel", 1.0),
        ("channel", "elementwise", 0.0),
        ("channel", "elementwise", 1.0),
        ("channel", "elementwise", -0.5),
    ],
)
def test_fused_scale_shift_generalization(
    shape, dtype, scale_mode, shift_mode, scale_constant
):
    """Exercise other shapes without repeating ignored broadcast constants."""
    from sgl_kernel_npu.operators.elementwise import fused_scale_shift

    inputs = generate_test_data(shape, dtype, scale_mode, shift_mode)
    executor = check_accuracy(
        fused_scale_shift, args=inputs, kwargs={"scale_constant": scale_constant}
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "tokens,hidden_size",
    [
        (63, 127),
        (64, 128),
        (65, 129),
        (79, 127),
        (80, 128),
        (81, 129),
        (95, 127),
        (96, 128),
        (97, 129),
        (111, 127),
        (112, 128),
        (113, 129),
        (127, 127),
        (128, 128),
        (129, 129),
    ],
)
@pytest.mark.parametrize("shift_mode", ["channel", "elementwise"])
def test_fused_scale_shift_tile_boundaries(tokens, hidden_size, shift_mode):
    """Check masked tails below/at/above all original autotune tile sizes."""
    from sgl_kernel_npu.operators.elementwise import fused_scale_shift

    inputs = generate_test_data(
        (1, tokens, hidden_size), torch.float32, "channel", shift_mode
    )
    executor = check_accuracy(fused_scale_shift, args=inputs)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "dtypes",
    [
        dtypes
        for dtypes in product((torch.float32, torch.float16, torch.bfloat16), repeat=3)
        if len(set(dtypes)) > 1
    ],
)
@pytest.mark.parametrize("scale_mode,shift_mode", modulation_cases)
def test_fused_scale_shift_mixed_dtypes(dtypes, scale_mode, shift_mode):
    """Explore mixed dtype permutations beyond required caller scenarios."""
    from sgl_kernel_npu.operators.elementwise import fused_scale_shift

    inputs = generate_test_data((2, 3, 129), torch.float32, scale_mode, shift_mode)
    inputs = tuple(value.to(dtype) for value, dtype in zip(inputs, dtypes))
    executor = check_accuracy(
        fused_scale_shift, args=inputs, kwargs={"scale_constant": -0.5}
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape,scale_mode,shift_mode,expected_factor",
    [
        pytest.param((2, 3, 129), "scalar", "channel", 1.0, id="scalar-channel"),
        pytest.param((2, 3, 129), "channel", "scalar", 1.00390625, id="channel-scalar"),
        pytest.param((2, 3, 1), "scalar", "scalar", 1.0, id="one-channel-scalar"),
        pytest.param((2, 3, 1), "channel", "channel", 1.0, id="one-channel-broadcast"),
        pytest.param(
            (2, 3, 1), "channel", "elementwise", -0.4990234375, id="one-channel-full"
        ),
        pytest.param((1, 1, 129), "channel", "channel", -0.4990234375, id="single-row"),
        pytest.param((1, 1, 1), "scalar", "scalar", -0.4990234375, id="single-element"),
    ],
)
def test_fused_scale_shift_bf16_scale_rounding_extra(
    x_dtype, shape, scale_mode, shift_mode, expected_factor
):
    """Check uncommon broadcasts, H=1 and overlapping element counts."""
    _check_bf16_scale_rounding(shape, x_dtype, scale_mode, shift_mode, expected_factor)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,shift_mode,zero_input",
    [
        ((2, 3, 129), "channel", False),
        ((2, 3, 129), "elementwise", False),
        ((1, 1, 129), "channel", False),
        ((1, 1, 1), "scalar", False),
        ((2, 3, 129), "elementwise", True),
    ],
)
def test_fused_scale_shift_output_contract(
    shape,
    dtype,
    shift_mode,
    zero_input,
):
    """Use exact values to check branch precedence, constants and no mutation."""
    if get_skip_acc_check() is True:
        pytest.skip("Output checks are disabled by --skip-acc-check=true")

    from sgl_kernel_npu.operators.elementwise import fused_scale_shift

    x, scale, shift = generate_test_data(shape, dtype, "channel", shift_mode)
    x.fill_(0.0 if zero_input else 2.0)
    scale.fill_(0.5)
    shift.fill_(0.25)
    snapshots = tuple(value.clone() for value in (x, scale, shift))
    impl, _ = get_test_impls()
    require_test_impl(fused_scale_shift, impl)
    result = fused_scale_shift(x, scale, shift, -0.5, impl=impl)

    # Equal element counts select the elementwise path, including L=B=1
    # where a channel shift is also a full-input shift.
    expected = 0.25 if zero_input or shift.numel() == x.numel() else 3.25
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert result.device == x.device
    assert result.is_contiguous()
    assert result.untyped_storage().data_ptr() not in [
        value.untyped_storage().data_ptr() for value in (x, scale, shift)
    ]
    torch.testing.assert_close(result, torch.full_like(x, expected), atol=0, rtol=0)
    for value, snapshot in zip((x, scale, shift), snapshots):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid_input,match",
    [
        ("x_not_tensor", "x must be a torch.Tensor"),
        ("scale_not_tensor", "scale must be a torch.Tensor"),
        ("shift_not_tensor", "shift must be a torch.Tensor"),
        ("x_rank", "x must have shape"),
        ("empty_batch", "x dimensions must be positive"),
        ("empty_tokens", "x dimensions must be positive"),
        ("empty_hidden", "x dimensions must be positive"),
        ("x_dtype", "x must have float32"),
        ("scale_dtype", "scale must have float32"),
        ("shift_dtype", "shift must have float32"),
        ("scale_device", "scale must be on x.device"),
        ("shift_device", "shift must be on x.device"),
        ("cpu_inputs", "x must be on an NPU"),
        ("x_stride", "x must be contiguous"),
        ("scale_stride", "scale must be contiguous"),
        ("shift_stride", "shift must be contiguous"),
        ("scale_size", "Scale size must be"),
        ("shift_size", "Shift size must be"),
        ("scalar_scale_full_shift", "Elementwise shift requires"),
        ("scalar_scale_single_row", "Elementwise shift requires"),
        ("constant_bool", "scale_constant must be a finite Python"),
        ("constant_tensor", "scale_constant must be a finite Python"),
        ("constant_nan", "scale_constant must be finite"),
        ("constant_inf", "scale_constant must be finite"),
    ],
)
def test_fused_scale_shift_invalid_input(
    invalid_input,
    match,
):
    """Reject unsafe metadata in the Triton wrapper before launching a kernel."""
    from sgl_kernel_npu.operators.elementwise import fused_scale_shift

    x, scale, shift = generate_test_data(
        (2, 3, 129), torch.float32, "channel", "channel"
    )
    scale_constant = 1.0
    if invalid_input == "x_not_tensor":
        x = None
    elif invalid_input == "scale_not_tensor":
        scale = None
    elif invalid_input == "shift_not_tensor":
        shift = None
    elif invalid_input == "x_rank":
        x = x.flatten(0, 1)
    elif invalid_input == "empty_batch":
        x = x[:0]
    elif invalid_input == "empty_tokens":
        x = torch.empty(2, 0, 129)
    elif invalid_input == "empty_hidden":
        x = torch.empty(2, 3, 0)
    elif invalid_input == "x_dtype":
        x = x.to(torch.int32)
    elif invalid_input == "scale_dtype":
        scale = scale.to(torch.int32)
    elif invalid_input == "shift_dtype":
        shift = shift.to(torch.int32)
    elif invalid_input == "scale_device":
        scale = scale.cpu()
    elif invalid_input == "shift_device":
        shift = shift.cpu()
    elif invalid_input == "cpu_inputs":
        x, scale, shift = x.cpu(), scale.cpu(), shift.cpu()
    elif invalid_input == "x_stride":
        x = torch.empty(2, 3, 258)[..., ::2]
    elif invalid_input == "scale_stride":
        scale = torch.empty(258)[::2]
    elif invalid_input == "shift_stride":
        shift = torch.empty(258)[::2]
    elif invalid_input == "scale_size":
        scale = torch.empty(128)
    elif invalid_input == "shift_size":
        shift = torch.empty(128)
    elif invalid_input == "scalar_scale_full_shift":
        scale, shift = torch.ones(1), torch.zeros_like(x)
    elif invalid_input == "scalar_scale_single_row":
        x, scale = x[:1, :1].contiguous(), torch.ones(1)
    elif invalid_input == "constant_bool":
        scale_constant = True
    elif invalid_input == "constant_tensor":
        scale_constant = torch.tensor(1.0)
    elif invalid_input == "constant_nan":
        scale_constant = float("nan")
    elif invalid_input == "constant_inf":
        scale_constant = float("inf")
    else:
        raise AssertionError(f"Unhandled invalid input: {invalid_input}")
    with pytest.raises(AssertionError, match=match):
        fused_scale_shift(x, scale, shift, scale_constant, impl="triton")
