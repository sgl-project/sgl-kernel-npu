"""A minimal registry: (operator name, impl name) -> callable."""

_IMPLS = {}
_OPERATOR_NAMES = {}


def register_operator(op_name):
    """Associate a public function object with its unique operator name."""
    def decorator(fn):
        if fn in _OPERATOR_NAMES:
            raise ValueError("Public operator function already registered: {}".format(fn))
        if op_name in _OPERATOR_NAMES.values():
            raise ValueError("Duplicate public operator name: {}".format(op_name))
        _OPERATOR_NAMES[fn] = op_name
        return fn

    return decorator


def get_op_name(operator):
    """Look up a registered public function; never infer names from paths."""
    if operator not in _OPERATOR_NAMES:
        raise ValueError("Unregistered public operator: {!r}".format(operator))
    return _OPERATOR_NAMES[operator]


def register_impl(op_name, impl):
    """Register a callable when its containing module is imported."""
    def decorator(fn):
        key = (op_name, impl)
        if key in _IMPLS:
            raise ValueError("Duplicate impl: {}".format(key))
        _IMPLS[key] = fn
        return fn

    return decorator


def list_impls_by_name(op_name):
    """Return sorted impl names registered for an operator.

    Import the operator's module first so its decorators have run.
    Returns an empty list for an unknown or not-yet-imported operator.
    Registration does not imply availability on the current device.
    """
    return sorted(name for op, name in _IMPLS if op == op_name)


def get_impl_by_name(op_name, impl):
    """Resolve an explicit impl; never silently fall back."""
    key = (op_name, impl)
    if key not in _IMPLS:
        available = list_impls_by_name(op_name)
        raise ValueError(
            "Unknown impl {!r} for {}; available: {}".format(
                impl, op_name, available
            )
        )
    return _IMPLS[key]


def get_impl(operator, impl):
    """Resolve an implementation using the registered public function."""
    return get_impl_by_name(get_op_name(operator), impl)


def list_impls(operator):
    """List registered choices for a public function, not device availability."""
    return list_impls_by_name(get_op_name(operator))
