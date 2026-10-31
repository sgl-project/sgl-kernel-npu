"""Public-interface coverage for both scale/shift kernel paths.

SGLang's multimodal_gen/runtime/layers/elementwise.py::MulAdd.forward_npu
passes [B, L, H] tensors and scale_constant=k. The layernorm.py NPU path
flattens scalar/channel modulation before calling this operator. Wan's
config uses H = 40 heads * 128 = 5120; the token counts below are synthetic
video-token tiles, not runtime traces. The legacy test used [3, 37440, 5120];
we retain its width and batch coverage without that multi-GB allocation.
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

perf_shape = [(1, 128, 5120), (1, 257, 5120)]

modulation_cases = [
    pytest.param("scalar", "scalar", id="scalar-scale-scalar-shift"),
    pytest.param("scalar", "channel", id="scalar-scale-channel-shift"),
    pytest.param("channel", "scalar", id="channel-scale-scalar-shift"),
    pytest.param("channel", "channel", id="channel-scale-channel-shift"),
    pytest.param("channel", "elementwise", id="channel-scale-elementwise-shift"),
]


def generate_test_data(shape, dtype, scale_mode, shift_mode, seed=42):
    """Generate finite NPU data; modest values limit low-precision roundoff."""
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 128, 5120), id="wan-width-video-token-tile"),
        pytest.param((1, 257, 5120), id="wan-width-video-token-tail"),
        pytest.param((3, 5, 5120), id="legacy-batch-and-width"),
        pytest.param((2, 3, 1), id="minimum-hidden-multiple-batches"),
        pytest.param((1, 7, 129), id="odd-rows-hidden-tail"),
    ],
)
@pytest.mark.parametrize("scale_mode,shift_mode", modulation_cases)
@pytest.mark.parametrize("scale_constant", [0.0, 1.0, -0.5])
def test_fused_scale_shift(
    shape,
    dtype,
    scale_mode,
    shift_mode,
    scale_constant,
):
    from sgl_kernel_npu.operators.norm import fused_scale_shift

    inputs = generate_test_data(shape, dtype, scale_mode, shift_mode)
    executor = check_accuracy(
        fused_scale_shift,
        args=inputs,
        kwargs={"scale_constant": scale_constant},
    )
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and dtype == torch.float32
            and scale_mode == "channel"
            and shift_mode in ("channel", "elementwise")
            and scale_constant == 1.0
        ),
    )


@pytest.mark.parametrize(
    "tokens,hidden_size",
    [
        # Below/at/above each row tile in the two original autotune lists;
        # 127/128/129 also exercise the hidden-channel tile and its tails.
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
def test_fused_scale_shift_tile_boundaries(
    tokens,
    hidden_size,
    shift_mode,
):
    from sgl_kernel_npu.operators.norm import fused_scale_shift

    inputs = generate_test_data(
        (1, tokens, hidden_size), torch.float32, "channel", shift_mode
    )
    executor = check_accuracy(fused_scale_shift, args=inputs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "dtypes",
    list(product((torch.float32, torch.float16, torch.bfloat16), repeat=3)),
)
@pytest.mark.parametrize("shift_mode", ["scalar", "channel", "elementwise"])
def test_fused_scale_shift_mixed_dtypes(
    dtypes,
    shift_mode,
):
    from sgl_kernel_npu.operators.norm import fused_scale_shift

    inputs = generate_test_data((2, 3, 129), torch.float32, "channel", shift_mode)
    inputs = tuple(value.to(dtype) for value, dtype in zip(inputs, dtypes))
    executor = check_accuracy(
        fused_scale_shift, args=inputs, kwargs={"scale_constant": -0.5}
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 129), (1, 1, 129), (1, 1, 1)])
@pytest.mark.parametrize("shift_mode", ["scalar", "channel", "elementwise"])
@pytest.mark.parametrize("zero_input", [False, True])
def test_fused_scale_shift_output_contract(
    shape,
    dtype,
    shift_mode,
    zero_input,
):
    """Use exact values to check branch precedence, constants and no mutation."""
    if get_skip_acc_check() is True:
        pytest.skip("Output checks are disabled by --skip-acc-check=true")

    from sgl_kernel_npu.operators.norm import fused_scale_shift

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
    from sgl_kernel_npu.operators.norm import fused_scale_shift

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
