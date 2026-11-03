"""Generic binding discovery, cached selection and schema checks.

Operator modules supply the binding specifications. Choices are cached before
execution; a failed kernel is never retried with a different backend.
"""

import functools
import importlib
import os
import re
from pathlib import Path
from typing import NamedTuple

import torch

from ._cann_ops_transformer import get_op

_VERSION_RE = re.compile(
    r"^\s*(?:version|version_dir)\s*=\s*(\d+\.\d+\.\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


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
def detect_cann_version() -> str:
    """Read an exact release from the initialized SDK before selecting bindings."""
    sdk = os.environ.get("ASCEND_HOME_PATH")
    if not sdk:
        raise RuntimeError("Initialize CANN: ASCEND_HOME_PATH is missing")

    sdk_path = Path(sdk)
    if not sdk_path.is_dir():
        raise RuntimeError(f"ASCEND_HOME_PATH does not exist: {sdk}")

    candidates = (sdk_path / "version.cfg", sdk_path / "compiler" / "version.info")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"Cannot read CANN metadata {path}: {exc}") from exc
        match = _VERSION_RE.search(text)
        if match:
            return match.group(1)

    raise RuntimeError(
        "Cannot read an exact CANN release from SDK version metadata "
        f"(looked in: {', '.join(map(str, candidates))})"
    )


@functools.lru_cache(maxsize=1)
def cann_backend():
    """Select the supported SDK release family without probing a fallback."""
    version = detect_cann_version()
    release = tuple(map(int, version.split(".")))[:2]
    if release == (9, 1):
        return "legacy"
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
