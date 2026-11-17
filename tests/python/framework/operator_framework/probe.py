"""Test-only integration pattern; this is not a production dispatcher.

Loaded beneath each private package so registration uses the real isolated
registry. Backend names are arbitrary labels, not actual implementation languages.
"""

from importlib import import_module

from .operators.utils.registry import get_impl_by_name, register_impl, register_operator

OP_NAME = "framework.dispatch_probe"
_backend_a_fn = None
_backend_b_fn = None


@register_impl(OP_NAME, "auto")
@register_impl(OP_NAME, "backend_a")
def _backend_a(x, offset=1, *, scale=1):
    global _backend_a_fn
    if _backend_a_fn is None:
        _backend_a_fn = import_module(f"{__package__}._probe_backends.backend_a").run
    return _backend_a_fn(x, offset, scale=scale)


@register_impl(OP_NAME, "backend_b")
def _backend_b(x, offset=1, *, scale=1):
    global _backend_b_fn
    if _backend_b_fn is None:
        _backend_b_fn = import_module(f"{__package__}._probe_backends.backend_b").run
    return _backend_b_fn(x, offset, scale=scale)


@register_operator(OP_NAME)
def probe(x, offset=1, scale=1, *, impl="auto"):
    """Route a lightweight operation, x * scale + offset, via the real registry."""
    return get_impl_by_name(OP_NAME, impl)(x, offset, scale=scale)
