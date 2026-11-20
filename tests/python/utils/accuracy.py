"""Shared accuracy checks for outputs from different backends."""

import torch

from .common import (
    get_skip_acc_check,
    make_executor,
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
        actual, expected = result.detach(), result_ref.detach()
        if str(expected.dtype) in ("torch.float8_e4m3fn", "torch.float8_e5m2"):
            # Transfer the encoding before comparing values with the requested
            # tolerance. The tested device may lack FP8 comparison kernels.
            actual = actual.view(torch.uint8).cpu().view(actual.dtype).float()
            expected = expected.view(torch.uint8).cpu().view(expected.dtype).float()
        torch.testing.assert_close(
            actual,
            expected,
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


def _clone_buffer(value):
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_buffer(item) for item in value)
    return value.clone()


def check_accuracy(
    operator,
    args=(),
    *,
    kwargs=None,
    impl=None,
    impl_ref=None,
    standard=None,
    skip_acc_check=False,
    mutable_kwargs=(),
    result_transform=None,
):
    """Validate two public-interface implementations and return an executor.

    The operator must accept impl as a keyword. List any writable tensor or
    tensor sequence passed by name in mutable_kwargs. The reference receives
    independent copies of these buffers. Their updated contents and the return
    values are compared. Other inputs are shared and must be read-only.
    A result_transform can select defined outputs or normalize permitted
    variations (for example sorting TopK indices with unspecified output order).
    Later benchmark iterations reuse the DUT inputs and buffers.
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
    executor = make_executor(
        operator, args=args, kwargs=kwargs, impl=impl, impl_ref=impl_ref
    )
    impl, impl_ref = executor.impl, executor.impl_ref
    if skip_acc_check:
        require_test_impl(operator, impl)
        print(f"Accuracy check skipped: {operator.__name__} [{impl}]")
        return executor
    require_distinct_impls(operator, impl, impl_ref)
    ref_kwargs = {**executor.keywords, "impl": impl_ref}
    for name in mutable_kwargs:
        ref_kwargs[name] = _clone_buffer(ref_kwargs[name])
    result_ref = operator(*executor.args, **ref_kwargs)
    result = executor()
    if result_transform is not None:
        result = result_transform(result)
        result_ref = result_transform(result_ref)
    assert_accuracy(result, result_ref, standard)
    for name in mutable_kwargs:
        assert_accuracy(executor.keywords[name], ref_kwargs[name], standard, path=name)
    executor.accuracy_checked = True
    return executor
