"""Generation-aware registry: (operator name, impl name) -> callable."""

from threading import RLock

from .generation import Generation, get_hardware


class ImplementationNotFoundError(ValueError):
    """No implementation with this name was declared."""


class UnsupportedGenerationError(RuntimeError):
    """The implementation exists, but not for this process's hardware."""

    def __init__(self, op_name, impl, hardware, supported_generations):
        self.op_name = op_name
        self.impl = impl
        self.hardware = hardware
        self.supported_generations = supported_generations
        supported = sorted(g.value for g in supported_generations)
        super().__init__(
            "{} [impl={}] does not support generation={}; supported generations={}".format(
                op_name, impl, hardware.generation.value, supported
            )
        )


_IMPLS = {}
# Every declared (op_name, impl) keeps its remaining unregistered generations,
# including an empty set once all known generations have an entry.
_UNSUPPORTED_IMPLS = {}
_OPERATOR_NAMES = {}

# All visible cards must belong to one generation. Import requires an NPU;
# detection failures propagate here rather than during operator execution.
_HARDWARE = get_hardware()
_LOCK = RLock()


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
    with _LOCK:
        name = _OPERATOR_NAMES.get(operator)
    if name is None:
        raise ValueError("Unregistered public operator: {!r}".format(operator))
    return name


def register_impl(op_name, impl, *, supported_generations=None, excluded_generations=None):
    """Classify support using import-time hardware, without importing the backend.

    Omit both lists to support all known generations. Only one may be supplied;
    members must be Generation values. Empty support means no generations.
    The same (op_name, impl) may have multiple entries with disjoint support.
    """
    if supported_generations is not None and excluded_generations is not None:
        raise ValueError("Set only supported_generations or excluded_generations")
    known = frozenset(Generation)
    values = supported_generations if supported_generations is not None else excluded_generations
    if values is not None:
        values = frozenset(values)
        if any(not isinstance(value, Generation) for value in values):
            raise ValueError("Generation lists must contain Generation enum members")
    supported = values if supported_generations is not None else known - (values or frozenset())

    def decorator(fn):
        key = (op_name, impl)
        with _LOCK:
            unsupported = _UNSUPPORTED_IMPLS.get(key, known)
            overlap = (known - unsupported) & supported
            if overlap:
                raise ValueError(
                    "Duplicate impl: {}; overlapping generations={}".format(
                        key, sorted(g.value for g in overlap)
                    )
                )
            _UNSUPPORTED_IMPLS[key] = unsupported - supported
            if _HARDWARE.generation in supported:
                _IMPLS[key] = fn
        return fn

    return decorator


def list_impls_by_name(op_name):
    """Return sorted impl names declared for an operator.

    Import the operator's module first so its decorators have run. Returns an
    empty list for an unknown or not-yet-imported operator. Registration does
    not imply availability on the current device.
    """
    with _LOCK:
        keys = tuple(_UNSUPPORTED_IMPLS)
    return sorted(name for op, name in keys if op == op_name)


def get_impl_by_name(op_name, impl):
    """Resolve an explicit impl after its module import completes; no fallback."""
    key = (op_name, impl)
    # Read both tables under the lock, each exactly once: a registration that
    # runs concurrently must not be observed half-applied (present in one table
    # but not the other). _LOCK is reentrant, so the fallback call below is safe.
    with _LOCK:
        fn = _IMPLS.get(key)
        uncovered = _UNSUPPORTED_IMPLS.get(key)
    if fn is not None:
        return fn
    if uncovered is not None:
        raise UnsupportedGenerationError(op_name, impl, _HARDWARE, frozenset(Generation) - uncovered)
    raise ImplementationNotFoundError(
        "Unknown impl {!r} for {}; declared: {}".format(impl, op_name, list_impls_by_name(op_name))
    )


def get_impl(operator, impl):
    """Resolve an implementation using the registered public function."""
    return get_impl_by_name(get_op_name(operator), impl)


def list_impls(operator):
    """List registered choices for a public function, not device availability.

    The result may include implementations that are declared but unavailable on
    this device (for example ``ascendc`` on a generation without an entry
    point); use :func:`get_impl_by_name` to probe actual availability.
    """
    return list_impls_by_name(get_op_name(operator))
