"""A minimal registry: (operator name, implementation name) -> callable."""

_IMPLS = {}


def register_impl(op_name, implementation):
    """Register a callable when its containing module is imported."""
    def decorator(fn):
        key = (op_name, implementation)
        if key in _IMPLS:
            raise ValueError("Duplicate implementation: {}".format(key))
        _IMPLS[key] = fn
        return fn

    return decorator


def list_impls(op_name):
    """Return sorted implementation names registered for an operator.

    Import the operator's module first so its decorators have run.
    Returns an empty list for an unknown or not-yet-imported operator.
    Registration does not imply availability on the current device.
    """
    return sorted(name for op, name in _IMPLS if op == op_name)


def get_impl(op_name, implementation):
    """Resolve an explicit implementation; never silently fall back."""
    key = (op_name, implementation)
    if key not in _IMPLS:
        available = list_impls(op_name)
        raise ValueError(
            "Unknown implementation {!r} for {}; available: {}".format(
                implementation, op_name, available
            )
        )
    return _IMPLS[key]
