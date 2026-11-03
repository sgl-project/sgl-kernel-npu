"""Generic binding discovery, cached selection and schema checks.

Operator modules supply the binding specifications. Choices are cached before
execution; a failed kernel is never retried with a different backend.
"""

import functools
import importlib
import os
from pathlib import Path
from typing import NamedTuple

import torch

from ._cann_ops_transformer import get_op


class OperatorUnavailableError(RuntimeError):
    """The installed operator libraries cannot implement the requested API."""


class OperatorFamily(NamedTuple):
    backend: str
    ops: tuple
    names: tuple

    @property
    def legacy(self):
        return self.backend == "legacy"


@functools.lru_cache(maxsize=1)
def backend_preference():
    value = os.environ.get("SGL_KERNEL_NPU_TRANSFORMER_BACKEND", "auto")
    if value not in ("auto", "cann_ops_transformer", "legacy"):
        raise ValueError(
            "SGL_KERNEL_NPU_TRANSFORMER_BACKEND must be auto, "
            "cann_ops_transformer, or legacy"
        )
    return value


@functools.lru_cache(maxsize=1)
def has_released_ops():
    try:
        importlib.import_module("cann_ops_transformer")
    except ModuleNotFoundError as error:
        if error.name != "cann_ops_transformer":
            raise  # A broken installed package is not an absent backend.
        return False
    return True


def prefer_legacy_bindings():
    preference = backend_preference()
    return preference == "legacy" or (preference == "auto" and not has_released_ops())


@functools.lru_cache(maxsize=1)
def _load_legacy_library():
    # Existing deployments may already register custom ops themselves. Loading
    # a standalone library is optional and belongs to the kernel package.
    path = os.environ.get("SGL_KERNEL_NPU_LEGACY_OPS_SO")
    if path:
        library = Path(path).expanduser().resolve(strict=True)
        if not library.is_file():
            raise OperatorUnavailableError(
                f"Legacy operator library is not a file: {library}"
            )
        torch.ops.load_library(str(library))


@functools.lru_cache(maxsize=None)
def optional_op(namespace, name):
    if namespace == "cann_ops_transformer":
        if not has_released_ops():
            return None
        if not hasattr(torch.ops.cann_ops_transformer, name):
            return None
        return get_op(name)
    if namespace == "custom":
        _load_legacy_library()
    return getattr(getattr(torch.ops, namespace), name, None)


def _lookup(specs):
    return tuple(optional_op(namespace, name) for namespace, name in specs)


@functools.lru_cache(maxsize=None)
def select_family(released, legacy, *, prefer_legacy=False, required_arguments=()):
    """Resolve a whole metadata/compute/writer family, never individual halves."""
    preference = backend_preference()
    order = (
        ("legacy", "cann_ops_transformer")
        if prefer_legacy
        else ("cann_ops_transformer", "legacy")
    )
    if preference != "auto":
        order = (preference,)
    missing = []
    for backend in order:
        specs = released if backend == "cann_ops_transformer" else legacy
        ops = _lookup(specs)
        if all(op is not None for op in ops):
            unsupported = []
            for op in ops:
                names = schema_arguments(op)
                if names is not None:
                    unsupported.extend(
                        name for name in required_arguments if name not in names
                    )
            if not unsupported:
                return OperatorFamily(backend, ops, specs)
            missing.append(f"{backend} arguments {sorted(set(unsupported))}")
        missing.extend(
            f"{ns}.{name}" for (ns, name), op in zip(specs, ops) if op is None
        )
    raise OperatorUnavailableError(
        "No complete compatible operator family is registered. Missing: "
        + ", ".join(missing)
        + ". Register the released CANN extension or the original custom/native "
        "libraries before the first call. Backend selection cannot change after "
        "KV allocation or graph capture."
    )


def schema_arguments(op):
    if isinstance(op, torch._ops.OpOverloadPacket):
        return {arg.name for arg in op.default._schema.arguments}
    return None


def checked_kwargs(op, kwargs, *, defaults=None):
    """Drop only absent, semantically unused defaults in an older schema."""
    names = schema_arguments(op)
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
