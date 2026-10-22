"""Shared accuracy checks for outputs from different backends."""

from functools import partial

import torch

from .common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)
from .standards import get_standard


def assert_accuracy(result, result_ref, standard=None, *, path="output"):
    """Compare nested tensors, tuples/lists, dicts and Python scalar values.

    standard maps dtype strings to AccuracyStandard overrides for Tensor leaves.
    Missing keys use defaults; integer Tensors default to exact comparison.
    Python integers, booleans, strings and None compare exactly.
    Unsupported output types raise TypeError instead of silently passing.
    """
    if isinstance(result, torch.Tensor) or isinstance(result_ref, torch.Tensor):
        assert isinstance(result, torch.Tensor) and isinstance(
            result_ref, torch.Tensor
        ), f"{path}: Tensor/non-Tensor mismatch"
        assert result.shape == result_ref.shape, f"{path}: shape mismatch"
        assert result.dtype == result_ref.dtype, f"{path}: dtype mismatch"
        assert result.device == result_ref.device, f"{path}: device mismatch"
        tolerance = get_standard(str(result_ref.dtype), standard)
        if tolerance is not None:
            atol, rtol = tolerance.atol, tolerance.rtol
        elif result_ref.is_floating_point() or result_ref.is_complex():
            raise TypeError(f"{path}: no accuracy standard for {result_ref.dtype}")
        else:
            atol, rtol = 0, 0
        torch.testing.assert_close(
            result.detach(),
            result_ref.detach(),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            msg=lambda message: f"{path}: {message}",
        )
        return

    assert type(result) is type(
        result_ref
    ), f"{path}: type mismatch ({type(result).__name__} vs {type(result_ref).__name__})"
    if isinstance(result_ref, (tuple, list)):
        assert len(result) == len(result_ref), f"{path}: length mismatch"
        for index, (value, other) in enumerate(zip(result, result_ref)):
            assert_accuracy(value, other, standard, path=f"{path}[{index}]")
    elif isinstance(result_ref, dict):
        assert result.keys() == result_ref.keys(), f"{path}: dictionary keys mismatch"
        for key in result_ref:
            assert_accuracy(
                result[key], result_ref[key], standard, path=f"{path}[{key!r}]"
            )
    elif isinstance(result_ref, (float, complex)):
        tolerance = get_standard("torch.float32", standard)
        torch.testing.assert_close(
            result,
            result_ref,
            atol=tolerance.atol,
            rtol=tolerance.rtol,
            equal_nan=False,
            msg=lambda message: f"{path}: {message}",
        )
    elif result_ref is None or isinstance(result_ref, (bool, int, str, bytes)):
        assert (
            result == result_ref
        ), f"{path}: value mismatch ({result!r} vs {result_ref!r})"
    else:
        raise TypeError(f"{path}: unsupported output type {type(result_ref).__name__}")


def check_accuracy(
    operator,
    args=(),
    *,
    kwargs=None,
    impl=None,
    impl_ref=None,
    standard=None,
    skip_acc_check=False,
):
    """Validate two public-interface implementations and return an executor.

    The operator must accept impl as a keyword and not mutate its inputs.
    args/kwargs are shared by both calls and later benchmark iterations.
    Corresponding output Tensors must be on the same device, regardless of type.
    No performance measurement runs here; errors propagate to the test.
    Each omitted implementation name comes from the current test configuration.
    Explicit names override that side only. The executor carries impl/impl_ref
    metadata with the effective choices.
    standard optionally maps dtype strings to AccuracyStandard overrides;
    omitted dtypes retain their default tolerances.
    skip_acc_check skips both accuracy calls and comparison (default: False).
    An explicit CLI true/false overrides the code setting. The returned executor
    remains usable for profiling; the reference implementation is not required.
    """
    cli_skip = get_skip_acc_check()
    if cli_skip is not None:
        skip_acc_check = cli_skip
    if not isinstance(skip_acc_check, bool):
        raise TypeError("skip_acc_check must be bool")
    if impl is None or impl_ref is None:
        default_impl, default_impl_ref = get_test_impls()
        if impl is None:
            impl = default_impl
        if impl_ref is None:
            impl_ref = default_impl_ref
    kwargs = dict(kwargs or {})
    if "impl" in kwargs:
        raise ValueError(
            "Select impl via check_accuracy or test configuration, not operator kwargs"
        )
    args = tuple(args)
    executor = partial(operator, *args, **kwargs, impl=impl)
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = False
    if skip_acc_check:
        require_test_impl(operator, impl)
        print(f"Accuracy check skipped: {operator.__name__} [{impl}]")
        return executor
    require_distinct_impls(operator, impl, impl_ref)
    result_ref = operator(*args, **kwargs, impl=impl_ref)
    result = executor()
    assert_accuracy(result, result_ref, standard)
    executor.accuracy_checked = True
    return executor
