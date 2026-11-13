import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(32, 2048, 4, 1), (4, 4096, 4, 4)]


def generate_data(shape, dtype, has_bias, speculative):
    batch, dim, width, steps = shape
    torch.manual_seed(42)
    state_len = width - 1 + steps - 1
    x = torch.randn(batch, steps, dim, dtype=dtype).transpose(1, 2)
    if not speculative:
        x = x.squeeze(-1)
    state = torch.randn(batch + 1, state_len, dim, dtype=dtype).transpose(1, 2)
    weight = torch.randn(dim, width, dtype=dtype)
    bias = torch.randn(dim, dtype=dtype) if has_bias else None
    indices = torch.arange(batch, dtype=torch.int32).flip(0)
    accepted = torch.ones(batch, dtype=torch.int32) if speculative else None
    intermediate = (
        torch.full((batch + 1, steps, dim, width - 1), -99, dtype=dtype)
        if speculative
        else None
    )
    return dict(
        x=x,
        conv_state=state,
        weight=weight,
        bias=bias,
        activation="swish",
        conv_state_indices=indices,
        num_accepted_tokens=accepted,
        intermediate_conv_window=intermediate,
    )


def _assert_original_standard(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    if delta.max() <= 1e-6:
        return
    ratio = delta.square().mean().sqrt() / (
        expected.float().square().mean().sqrt() + 1e-8
    )
    assert_accuracy(
        ratio,
        torch.zeros_like(ratio),
        standard={"torch.float32": AccuracyStandard(1e-3, 0)},
    )


@pytest.mark.parametrize(
    "shape,has_bias",
    [
        ((16, 1024, 3, 1), True),
        ((32, 1024, 3, 1), False),
        ((32, 2048, 3, 1), True),
        ((64, 2048, 3, 1), False),
        ((16, 1024, 4, 1), True),
        ((32, 1024, 4, 1), False),
        ((32, 2048, 4, 1), True),
        ((64, 2048, 4, 1), False),
    ],
)
def test_decode(shape, has_bias):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_update

    kwargs = generate_data(shape, torch.float16, has_bias, False)
    _check(causal_conv1d_update, kwargs, shape, original_standard=True)


def _check(operator, kwargs, shape, original_standard):
    impl, impl_ref = get_test_impls()
    executor = make_executor(operator, kwargs=kwargs)
    if not get_skip_acc_check():
        require_distinct_impls(operator, impl, impl_ref)
        reference = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        expected = operator(**reference, impl=impl_ref)
        actual = executor()
        if kwargs["num_accepted_tokens"] is not None:
            assert actual.data_ptr() == kwargs["x"].data_ptr()
        if original_standard:
            _assert_original_standard(actual, expected)
        else:
            assert_accuracy(actual, expected)
        assert_accuracy(
            kwargs["conv_state"],
            reference["conv_state"],
            standard={str(kwargs["conv_state"].dtype): AccuracyStandard(0, 0)},
        )
        if kwargs["intermediate_conv_window"] is not None:
            assert_accuracy(
                kwargs["intermediate_conv_window"],
                reference["intermediate_conv_window"],
                standard={str(kwargs["conv_state"].dtype): AccuracyStandard(0, 0)},
            )
        executor.accuracy_checked = True
    # State and input restoration are included in this stateful end-to-end measurement.
    initial = generate_data(
        shape,
        kwargs["x"].dtype,
        kwargs["bias"] is not None,
        kwargs["num_accepted_tokens"] is not None,
    )

    def run():
        kwargs["x"].copy_(initial["x"])
        kwargs["conv_state"].copy_(initial["conv_state"])
        return executor()

    run.func = executor.func
    run.impl, run.impl_ref, run.accuracy_checked = (
        executor.impl,
        executor.impl_ref,
        executor.accuracy_checked,
    )
    check_performance(run, enable_perf=shape in perf_shape)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1024, 4, 4), id="speculative-four-tokens"),
        pytest.param((4, 4096, 4, 4), id="gdn-target-verify"),
        pytest.param((3, 128, 3, 2), id="small-width-three"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_speculative(shape, dtype):
    from sgl_kernel_npu.operators.mamba import causal_conv1d_update

    kwargs = generate_data(shape, dtype, True, True)
    _check(causal_conv1d_update, kwargs, shape, original_standard=False)
