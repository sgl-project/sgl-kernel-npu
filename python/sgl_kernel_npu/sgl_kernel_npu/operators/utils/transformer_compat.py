"""Bundled binding loading, cached selection and schema checks.

Operator modules supply the binding specifications. Choices are cached before
execution; a failed kernel is never retried with a different backend.
"""

import functools

import torch

from .cann import detect_cann_version


class OperatorUnavailableError(RuntimeError):
    """The installed operator libraries cannot implement the requested API."""


@functools.lru_cache(maxsize=1)
def cann_backend():
    """Select the supported SDK backend without probing a fallback."""
    version = detect_cann_version()
    release = tuple(map(int, version.split(".")))[:2]
    if release == (9, 1):
        return "cann_9_1"
    if release == (9, 2):
        return "cann_ops_transformer_cann920"
    raise OperatorUnavailableError(
        f"Unsupported CANN release {version}; Transformer operators support "
        "CANN 9.1.x and 9.2.x only"
    )


@functools.lru_cache(maxsize=None)
def require_op(namespace, name, *, backend=None, required_arguments=()):
    """Load one bundled binding and validate the requested SDK and schema."""
    if backend is not None and backend != cann_backend():
        raise OperatorUnavailableError(
            f"CANN {detect_cann_version()} selects {cann_backend()}, not {backend}"
        )
    if namespace == "custom":
        # Importing the bundled extension registers the private Torch namespace.
        from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

        namespace = "sgl_cann_recipes_infer"
    elif namespace == "cann_ops_transformer":
        from ..._third_party.sgl_ops_transformer import ops  # noqa: F401

    elif namespace == "sgl_native_compressor":
        from ..._third_party import sgl_native_compressor  # noqa: F401
    elif namespace == "sgl_native_dsv4":
        from ..._third_party import sgl_native_dsv4  # noqa: F401
    op = getattr(getattr(torch.ops, namespace), name, None)
    if op is None:
        raise OperatorUnavailableError(
            f"Missing operator {namespace}.{name} in the installed sgl-kernel-npu "
            f"wheel for CANN {detect_cann_version()}"
        )
    arguments = get_operator_parameter_names(op)
    if arguments is not None:
        missing = set(required_arguments) - arguments
        if missing:
            raise OperatorUnavailableError(
                f"{namespace}.{name} is missing arguments: {', '.join(sorted(missing))}"
            )
    return op


def get_operator_parameter_names(op):
    if isinstance(op, torch._ops.OpOverloadPacket):
        return {arg.name for arg in op.default._schema.arguments}
    return None


def filter_supported_kwargs(op, kwargs, *, defaults=None):
    """Drop only absent, semantically unused defaults in an older schema."""
    names = get_operator_parameter_names(op)
    if names is None:
        return kwargs
    result = {}
    defaults = defaults or {}
    for name, value in kwargs.items():
        if name in names:
            result[name] = value
        elif value is None or (
            name in defaults
            and not isinstance(value, torch.Tensor)
            and value == defaults[name]
        ):
            continue
        else:
            raise OperatorUnavailableError(
                f"{op} does not support requested argument {name}={value!r}"
            )
    return result
