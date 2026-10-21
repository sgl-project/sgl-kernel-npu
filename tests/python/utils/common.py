"""Shared impl-selection checks."""

from contextlib import contextmanager
from contextvars import ContextVar

_IMPL_PAIR = ContextVar("test_impl_pair", default=("auto", "torch_native"))
_SKIP_ACC_CHECK = ContextVar("skip_accuracy_check", default=None)


def get_skip_acc_check():
    """Return the CLI override, or None when it was not provided."""
    return _SKIP_ACC_CHECK.get()


def get_test_impls():
    """Read the current test's implementation choices."""
    return _IMPL_PAIR.get()


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
