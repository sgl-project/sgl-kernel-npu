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

from .cann import detect_cann_version


class OperatorUnavailableError(RuntimeError):
    """The installed operator libraries cannot implement the requested API."""


class OperatorFamily(NamedTuple):
    backend: str
    ops: tuple
    names: tuple

    @property
    def legacy(self):
        return self.backend == "cann_9_1"


@functools.lru_cache(maxsize=1)
def cann_backend():
    """Select the supported SDK release family without probing a fallback."""
    version = detect_cann_version()
    release = tuple(map(int, version.split(".")))[:2]
    if release == (9, 1):
        return "cann_9_1"
    if release == (9, 2):
        return "cann_ops_transformer"
    raise OperatorUnavailableError(
        f"Unsupported CANN release {version}; Transformer operators support "
        "CANN 9.1.x and 9.2.x only"
    )


@functools.lru_cache(maxsize=1)
def has_released_ops():
    try:
        importlib.import_module("cann_ops_transformer")
    except ModuleNotFoundError as error:
        if error.name != "cann_ops_transformer":
            raise  # A broken installed package is not an absent backend.
        return False
    return True


@functools.lru_cache(maxsize=1)
def _load_legacy_library():
    # The legacy Python extension registers torch.ops.custom on import.
    # Only an absent package is optional; broken dependencies must surface.
    try:
        importlib.import_module("custom_ops")
    except ModuleNotFoundError as error:
        if error.name != "custom_ops":
            raise
    # A standalone registration library remains available for deployments
    # without the Python extension, or for additional custom bindings.
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
    if namespace == "cann_ops_transformer" and not has_released_ops():
        return None
    if namespace == "custom":
        _load_legacy_library()
    return getattr(getattr(torch.ops, namespace), name, None)


def _lookup(specs):
    return tuple(optional_op(namespace, name) for namespace, name in specs)


@functools.lru_cache(maxsize=None)
def select_family(released, legacy, *, required_arguments=()):
    """Require the complete family selected by the SDK version, without fallback."""
    backend = cann_backend()
    specs = released if backend == "cann_ops_transformer" else legacy
    ops = _lookup(specs)
    missing = [f"{ns}.{name}" for (ns, name), op in zip(specs, ops) if op is None]
    for op in ops:
        if op is None:
            continue
        names = schema_arguments(op)
        if names is not None:
            missing.extend(
                f"{op} argument {name}"
                for name in required_arguments
                if name not in names
            )
    if not missing:
        return OperatorFamily(backend, ops, specs)
    raise OperatorUnavailableError(
        f"CANN {detect_cann_version()} requires a complete compatible "
        f"{backend} operator family. Missing: "
        + ", ".join(missing)
        + ". Register the matching operator libraries before the first call; "
        "no other CANN backend is tried. Backend selection cannot change after "
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


@functools.lru_cache(maxsize=None)
def require_indexer_ops(backend):
    """Validate the selected ABI and its metadata/compute pair before execution.

    Explicit implementations cannot override the initialized SDK. This keeps
    opaque metadata and compute on the same ABI, without automatic fallback.
    """
    selected = cann_backend()
    if backend != selected:
        raise OperatorUnavailableError(
            f"CANN {detect_cann_version()} selects {selected}, not {backend}"
        )
    namespace = "custom" if backend == "cann_9_1" else "cann_ops_transformer"
    prefix = "npu_" if backend == "cann_9_1" else ""
    specs = tuple(
        (namespace, prefix + "quant_lightning_indexer" + suffix)
        for suffix in ("_metadata", "")
    )
    ops = _lookup(specs)
    missing = [f"{ns}.{name}" for (ns, name), op in zip(specs, ops) if op is None]
    if missing:
        raise OperatorUnavailableError(
            f"Missing {backend} indexer metadata/compute bindings: {', '.join(missing)}"
        )
    return ops
