"""Shared implementation selection, checks and executor construction."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

_IMPL_PAIR = ContextVar("test_impl_pair", default=("auto", "torch_native"))
_SKIP_ACC_CHECK = ContextVar("skip_accuracy_check", default=None)


def get_skip_acc_check():
    """Return the CLI override, or None when it was not provided."""
    return _SKIP_ACC_CHECK.get()


def get_test_impls():
    """Read the current test's implementation choices."""
    return _IMPL_PAIR.get()


def make_executor(
    operator,
    args=(),
    *,
    kwargs=None,
    impl=None,
    impl_ref=None,
    accuracy_checked=False,
):
    """Bind inputs and implementation metadata without executing the operator.

    Omitted implementation names use the current test configuration. Inputs are
    retained by reference, not copied or refreshed. This helper does not resolve
    registry entries or perform accuracy checks. Set accuracy_checked=True only
    after a successful manual comparison of these inputs and implementations.
    """
    if not isinstance(accuracy_checked, bool):
        raise TypeError("accuracy_checked must be bool")
    if impl is None or impl_ref is None:
        default_impl, default_impl_ref = get_test_impls()
        if impl is None:
            impl = default_impl
        if impl_ref is None:
            impl_ref = default_impl_ref
    kwargs = dict(kwargs or {})
    if "impl" in kwargs:
        raise ValueError(
            "Select impl via check_accuracy or make_executor or test configuration, "
            "not operator kwargs"
        )
    executor = partial(operator, *tuple(args), **kwargs, impl=impl)
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    return executor


@contextmanager
def use_impls(impl, impl_ref, *, skip_acc_check=None):
    """Scope settings to one test and restore them even if the test fails."""
    token = _IMPL_PAIR.set((impl, impl_ref))
    skip_token = _SKIP_ACC_CHECK.set(skip_acc_check)
    try:
        yield
    finally:
        _SKIP_ACC_CHECK.reset(skip_token)
        _IMPL_PAIR.reset(token)


def require_distinct_impls(operator, impl, impl_ref):
    """Reject unknown names and aliases of the same registered entry point."""
    from sgl_kernel_npu.operators.utils.registry import (
        UnsupportedGenerationError,
        get_impl,
    )

    unsupported = None
    try:
        actual_entry = get_impl(operator, impl)
    except UnsupportedGenerationError as error:
        unsupported = error
    # Reference errors take priority over a DUT generation skip.
    comparison_entry = get_impl(operator, impl_ref)
    if unsupported is not None:
        import pytest

        pytest.skip(str(unsupported))
    if actual_entry is comparison_entry:
        raise ValueError(
            f"{operator!r}: {impl!r} and {impl_ref!r} resolve to the "
            "same registered entry; choose two different implementations"
        )


def require_test_impl(operator, impl):
    """Resolve only the DUT when accuracy comparison is disabled."""
    from sgl_kernel_npu.operators.utils.registry import (
        UnsupportedGenerationError,
        get_impl,
    )

    try:
        return get_impl(operator, impl)
    except UnsupportedGenerationError as error:
        import pytest

        pytest.skip(str(error))
