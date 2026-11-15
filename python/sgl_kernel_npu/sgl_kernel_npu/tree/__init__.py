"""Lazy public API for the DSV4 DSpark tree operators.

Importing this package must **not** import Triton, ``torch_npu``, or any
kernel module.  Each public operator is resolved on first attribute access via
PEP 562 ``__getattr__``.  Wave 1 ships only the shared helpers and this lazy
surface; the kernel modules are added in later waves.

The canonical names mirror the AscendC bindings:

* :func:`compressor_tree_project`
* :func:`compressor_tree`
* :func:`quant_lightning_indexer_tree`
* :func:`sparse_attn_sharedkv_tree`

Short aliases (``tree_project``, ``tree_compressor``, ``tree_indexer``,
``tree_attention``) are also accepted.  Requesting a function whose module has
not been implemented yet raises a clear :class:`ImportError` instead of
silently returning a stub.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import _common

__all__ = [
    "compressor_tree_project",
    "compressor_tree",
    "quant_lightning_indexer_tree",
    "sparse_attn_sharedkv_tree",
]

common = _common

# canonical name -> (relative module, attribute)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "compressor_tree_project": (".compressor_tree_project", "compressor_tree_project"),
    "compressor_tree": (".compressor_tree", "compressor_tree"),
    "quant_lightning_indexer_tree": (
        ".quant_lightning_indexer_tree",
        "quant_lightning_indexer_tree",
    ),
    "sparse_attn_sharedkv_tree": (
        ".sparse_attn_sharedkv_tree",
        "sparse_attn_sharedkv_tree",
    ),
    # Short aliases used by benchmark/tests.
    "tree_project": (".compressor_tree_project", "compressor_tree_project"),
    "tree_compressor": (".compressor_tree", "compressor_tree"),
    "tree_indexer": (".quant_lightning_indexer_tree", "quant_lightning_indexer_tree"),
    "tree_attention": (".sparse_attn_sharedkv_tree", "sparse_attn_sharedkv_tree"),
}


def __getattr__(name: str) -> Any:
    """Resolve a public operator lazily; never import Triton at package import."""
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"sgl_kernel_npu.tree.{name} is unavailable: module "
            f"{module_name!r} has not been implemented. Wave 1 provides only "
            "the shared helpers and lazy API; see docs/tree_ops_semantics.md "
            "and tree/README.md for the planned kernels."
        ) from exc

    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"sgl_kernel_npu.tree.{name} is unavailable: module {module_name!r} "
            f"does not define {attr_name!r}."
        ) from exc

    # Cache the resolved object so subsequent lookups are cheap.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the public names in ``dir(sgl_kernel_npu.tree)``."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))
