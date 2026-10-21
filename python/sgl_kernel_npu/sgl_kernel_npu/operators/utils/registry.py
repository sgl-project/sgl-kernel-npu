"""Generation-aware registry: (operator name, impl name) -> callable."""

from threading import RLock

from .generation import Generation, get_hardware


class ImplementationNotFoundError(ValueError):
    """No implementation with this name was declared."""


class UnsupportedGenerationError(RuntimeError):
    """An implementation exists, but not for this process's hardware."""

    def __init__(self, op_name, impl, hardware, supported_generations):
        self.op_name = op_name
        self.impl = impl
        self.hardware = hardware
        self.supported_generations = supported_generations
        supported = sorted(g.value for g in supported_generations)
        super().__init__(
            f"{op_name} [impl={impl}] does not support SoC={hardware.soc_name}, "
            f"generation={hardware.generation.value}; supported generations={supported}"
        )

_IMPLS = {}
_DECLARATIONS = {}
_UNSUPPORTED_IMPLS = {}
_OPERATOR_NAMES = {}
# All visible cards must belong to one generation. Import requires an NPU;
# detection failures propagate here, not during operator execution.
_HARDWARE = get_hardware()
_LOCK = RLock()


def _classify(key, declaration, hardware):
    fn, supported = declaration
    if hardware.generation in supported:
        _IMPLS[key] = fn
    else:
        _UNSUPPORTED_IMPLS[key] = supported


def register_operator(op_name):
    """Associate a public function object with its unique operator name."""
    def decorator(fn):
        with _LOCK:
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


def register_impl(op_name, impl, *, supported_generations=None, excluded_generations=None):
    """Classify support using import-time hardware, without importing the backend.

    Omit both lists to support all known generations. Only one may be supplied;
    members must be Generation values. Empty support means no generations.
    """
    if supported_generations is not None and excluded_generations is not None:
        raise ValueError("Set only supported_generations or excluded_generations")
    known = frozenset(Generation)
    values = supported_generations if supported_generations is not None else excluded_generations
    if values is not None:
        values = frozenset(values)
        if any(not isinstance(value, Generation) for value in values):
            raise ValueError("Generation lists must contain Generation enum members")
    supported = (values if supported_generations is not None
                 else known - (values if values is not None else frozenset()))

    def decorator(fn):
        key = (op_name, impl)
        with _LOCK:
            if key in _DECLARATIONS:
                raise ValueError("Duplicate impl: {}".format(key))
            # Aliases of one callable must carry the same support declaration.
            for (name, _), (entry, generations) in _DECLARATIONS.items():
                if name == op_name and entry is fn and generations != supported:
                    raise ValueError("Aliases of an implementation must have identical support")
            _DECLARATIONS[key] = (fn, supported)
            _classify(key, (fn, supported), _HARDWARE)
        return fn

    return decorator


def list_impls_by_name(op_name):
    """Return sorted impl names registered for an operator.

    Import the operator's module first so its decorators have run.
    Returns an empty list for an unknown or not-yet-imported operator.
    Registration does not imply availability on the current device.
    """
    with _LOCK:
        keys = tuple(_DECLARATIONS)
    return sorted(name for op, name in keys if op == op_name)


def get_impl_by_name(op_name, impl):
    """Resolve an explicit impl after its module import completes; no fallback."""
    key = (op_name, impl)
    try:
        return _IMPLS[key]
    except KeyError:
        pass
    if key in _UNSUPPORTED_IMPLS:
        raise UnsupportedGenerationError(op_name, impl, _HARDWARE, _UNSUPPORTED_IMPLS[key])
    raise ImplementationNotFoundError(
        "Unknown impl {!r} for {}; declared: {}".format(
            impl, op_name, list_impls_by_name(op_name)
        )
    )


def get_impl(operator, impl):
    """Resolve an implementation using the registered public function."""
    return get_impl_by_name(get_op_name(operator), impl)


def list_impls(operator):
    """List registered choices for a public function, not device availability."""
    return list_impls_by_name(get_op_name(operator))
