"""Triton-free validation and tree-geometry helpers for DSV4 DSpark tree ops.

This module deliberately imports only :mod:`torch` and the standard library.
It must stay importable on a CPU-only host with no Triton, no ``torch_npu`` and
no built ``libsgl_kernel_npu.so``.  NPU-only helpers import their dependencies
lazily inside the function body.

Contract references use the pinned source commit ``8947f1f9f`` (see
``docs/tree_ops_semantics.md``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Sequence

import torch

__all__ = [
    "TreeContractError",
    "check_contiguous",
    "check_cu_arange",
    "check_dtype",
    "check_head_dim",
    "check_layout",
    "check_page_table",
    "check_rank",
    "check_ratio",
    "check_rope_dim",
    "check_same_device",
    "check_tensor",
    "check_tree_metadata",
    "ape_row",
    "tree_step",
    "tree_node",
    "paged_slot",
    "align_up",
    "ceil_div",
    "next_power_of_2",
    "floor_power_of_2",
    "pick_specialization",
    "get_num_vectorcores",
]

# Supported specializations (D6).  ratio 4 uses two interleaved halves; ratio
# 128 uses one.  head_dim is the projected per-half width.
_SUPPORTED_RATIOS = (4, 128)
_SUPPORTED_HEAD_DIMS = (128, 512)
_SUPPORTED_ROPE_DIMS = (64,)
_SUPPORTED_LAYOUTS_Q = ("TND",)
_SUPPORTED_LAYOUTS_KV = ("PA_ND",)


class TreeContractError(ValueError):
    """Raised when a caller violates a tree-op host contract."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_dtype(
    name: str,
    tensor: torch.Tensor,
    allowed: torch.dtype | Iterable[torch.dtype],
) -> torch.Tensor:
    """Raise :class:`TreeContractError` unless ``tensor.dtype`` is allowed."""
    if isinstance(allowed, torch.dtype):
        allowed = (allowed,)
    allowed = tuple(allowed)
    if tensor.dtype not in allowed:
        want = " | ".join(str(d) for d in allowed)
        raise TreeContractError(f"{name}: expected dtype {want}, got {tensor.dtype}")
    return tensor


def check_rank(
    name: str,
    tensor: torch.Tensor,
    rank: int | Iterable[int],
) -> torch.Tensor:
    """Raise unless ``tensor.dim()`` is one of the accepted ranks."""
    if isinstance(rank, int):
        rank = (rank,)
    rank = tuple(int(r) for r in rank)
    if tensor.dim() not in rank:
        want = " or ".join(str(r) for r in rank)
        raise TreeContractError(f"{name}: expected rank {want}, got {tensor.dim()}")
    return tensor


def check_contiguous(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Raise unless ``tensor`` is contiguous."""
    if not tensor.is_contiguous():
        raise TreeContractError(f"{name}: tensor must be contiguous")
    return tensor


def check_same_device(name: str, tensors: Sequence[torch.Tensor]) -> None:
    """Raise unless every tensor is on the same device as the first one."""
    if not tensors:
        return
    device = tensors[0].device
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.device != device:
            raise TreeContractError(
                f"{name}: tensors must be colocated; tensor[0] on {device}, "
                f"tensor[{i}] on {tensor.device}"
            )


def check_tensor(
    name: str,
    tensor: Any,
    *,
    dtype: torch.dtype | Iterable[torch.dtype] | None = None,
    rank: int | Iterable[int] | None = None,
    contiguous: bool = False,
) -> torch.Tensor:
    """Validate a torch tensor's type/rank/contiguity in one call."""
    if not isinstance(tensor, torch.Tensor):
        raise TreeContractError(f"{name}: expected a torch.Tensor, got {type(tensor)!r}")
    if dtype is not None:
        check_dtype(name, tensor, dtype)
    if rank is not None:
        check_rank(name, tensor, rank)
    if contiguous:
        check_contiguous(name, tensor)
    return tensor


def check_ratio(ratio: int) -> int:
    """Validate a compression ratio (4 or 128)."""
    ratio = int(ratio)
    if ratio not in _SUPPORTED_RATIOS:
        raise TreeContractError(
            f"ratio: expected one of {_SUPPORTED_RATIOS}, got {ratio}"
        )
    return ratio


def check_head_dim(head_dim: int) -> int:
    """Validate a projected head dimension (128 or 512)."""
    head_dim = int(head_dim)
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise TreeContractError(
            f"head_dim: expected one of {_SUPPORTED_HEAD_DIMS}, got {head_dim}"
        )
    return head_dim


def check_rope_dim(rope_dim: int) -> int:
    """Validate the RoPE dimension (64)."""
    rope_dim = int(rope_dim)
    if rope_dim not in _SUPPORTED_ROPE_DIMS:
        raise TreeContractError(
            f"rope_dim: expected {_SUPPORTED_ROPE_DIMS[0]}, got {rope_dim}"
        )
    return rope_dim


def check_layout(layout: str, *, name: str = "layout_q") -> str:
    """Validate a TND / PA_ND tree layout string (D4)."""
    text = str(layout)
    allowed = _SUPPORTED_LAYOUTS_Q if name == "layout_q" else _SUPPORTED_LAYOUTS_KV
    if text not in allowed:
        raise TreeContractError(f"{name}: expected {allowed}, got {text!r}")
    return text


def check_tree_metadata(
    name: str = "metadata",
    tensor: torch.Tensor | None = None,
    *,
    required: bool = False,
) -> torch.Tensor | None:
    """Validate the int32[1024] ABI metadata tensor; contents are ignored (D2)."""
    if tensor is None:
        if required:
            raise TreeContractError(f"{name}: metadata tensor is required")
        return None
    check_tensor(name, tensor, dtype=torch.int32, rank=1, contiguous=True)
    if tensor.numel() != 1024:
        raise TreeContractError(f"{name}: expected 1024 int32 entries, got {tensor.numel()}")
    return tensor


def check_cu_arange(name: str, cu: torch.Tensor, n: int) -> torch.Tensor:
    """Validate the project/indexer ``cu == arange(n + 1)`` one-query-per-node rule."""
    check_tensor(name, cu, dtype=torch.int32, rank=1, contiguous=True)
    n = int(n)
    if cu.numel() != n + 1:
        raise TreeContractError(f"{name}: expected {n + 1} entries, got {cu.numel()}")
    expected = torch.arange(n + 1, dtype=torch.int32, device=cu.device)
    if not torch.equal(cu, expected):
        raise TreeContractError(f"{name}: must equal arange({n + 1}) (one query per node)")
    return cu


def check_page_table(name: str, table: torch.Tensor) -> torch.Tensor:
    """Validate a contiguous int32 2-D page table."""
    check_tensor(name, table, dtype=torch.int32, rank=2, contiguous=True)
    return table


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _to_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TreeContractError(f"{name}: expected a scalar, got shape {tuple(value.shape)}")
        return int(value.item())
    return int(value)


def ape_row(position: int, ratio: int) -> int:
    """Return the APE row for a logical position: ``position % ratio``."""
    ratio = check_ratio(ratio)
    position = int(position)
    if position < 0:
        raise TreeContractError(f"position: expected >= 0, got {position}")
    return position % ratio


def tree_step(prefix: int, k: int, ratio: int = 4) -> int:
    """Ancestor step for logical compressed index ``k``.

    Mirrors ``tree_index.h``: ``step = (k + 1) * ratio - 1 - prefix``.
    A negative step means the logical index belongs to paged history, not the
    tree ancestor array.  (Ratio defaults to 4 because the indexer is C4-only;
    attention passes its own ``cmp_ratio``.)
    """
    ratio = check_ratio(ratio)
    return (int(k) + 1) * ratio - 1 - int(prefix)


def tree_node(
    ancestors: torch.Tensor | Sequence[Any],
    prefix: int,
    k: int,
    *,
    ratio: int = 4,
    row: int | None = None,
    node_count: int | None = None,
) -> int:
    """Resolve logical index ``k`` to a node id, or ``-1`` when absent.

    ``ancestors`` is root-first and may be a 1-D row, a 2-D ``[rows,width]``
    tensor/list, or a list of rows.  ``row`` is required for a 2-D input.
    ``node_count`` (when given) turns out-of-range node ids into ``-1``.
    """
    step = tree_step(prefix, k, ratio)
    if step < 0:
        return -1

    if isinstance(ancestors, torch.Tensor):
        if ancestors.dim() == 1:
            row_values = ancestors
        elif ancestors.dim() == 2:
            if row is None:
                raise TreeContractError("tree_node: row is required for 2-D ancestors")
            row_values = ancestors[int(row)]
        else:
            raise TreeContractError(
                f"tree_node: ancestors must be 1-D or 2-D, got rank {ancestors.dim()}"
            )
        width = int(row_values.numel())
        if step >= width:
            return -1
        node = int(row_values[step].item())
    else:
        if isinstance(ancestors, (list, tuple)) and ancestors and isinstance(ancestors[0], (list, tuple)):
            if row is None:
                raise TreeContractError("tree_node: row is required for nested ancestors")
            row_values = ancestors[int(row)]
        else:
            row_values = ancestors
        width = len(row_values)
        if step >= width:
            return -1
        node = _to_int(row_values[step], "ancestor")

    if node < 0:
        return -1
    if node_count is not None and node >= int(node_count):
        return -1
    return node


def paged_slot(table: torch.Tensor | Sequence[Any], req: int, pos: int, block: int) -> int:
    """Map a logical position to a physical slot via a page table.

    Returns ``-1`` when the position is negative, the page-table column is out
    of range, or the page id is negative.  ``table`` may be a tensor or a
    nested list; ``block`` is the number of rows per page.
    """
    block = int(block)
    if block <= 0:
        raise TreeContractError(f"block: expected > 0, got {block}")
    pos = int(pos)
    if pos < 0:
        return -1
    req = int(req)
    col = pos // block

    if isinstance(table, torch.Tensor):
        if table.dim() != 2:
            raise TreeContractError("paged_slot: table must be 2-D")
        if req < 0 or req >= table.size(0) or col >= table.size(1):
            return -1
        page = int(table[req, col].item())
    else:
        if req < 0 or req >= len(table) or col >= len(table[req]):
            return -1
        page = _to_int(table[req][col], "page")
    if page < 0:
        return -1
    return page * block + (pos % block)


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of ``alignment``."""
    alignment = int(alignment)
    if alignment <= 0:
        raise TreeContractError(f"alignment: expected > 0, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def ceil_div(a: int, b: int) -> int:
    """Integer ceiling division; ``b`` must be positive."""
    b = int(b)
    if b <= 0:
        raise TreeContractError(f"ceil_div: denominator must be > 0, got {b}")
    a = int(a)
    return -((-a) // b)


def next_power_of_2(x: int) -> int:
    """Smallest power of two >= ``x`` (``x >= 1``)."""
    x = int(x)
    if x < 1:
        raise TreeContractError(f"next_power_of_2: expected >= 1, got {x}")
    return 1 << (x - 1).bit_length()


def floor_power_of_2(x: int) -> int:
    """Largest power of two <= ``x`` (``x >= 1``)."""
    x = int(x)
    if x < 1:
        raise TreeContractError(f"floor_power_of_2: expected >= 1, got {x}")
    return 1 << (x.bit_length() - 1)


def pick_specialization(
    ratio: int,
    head_dim: int,
    rope_dim: int = 64,
) -> tuple[int, int, int, int]:
    """Return the Triton constexpr key ``(ratio, c, head_dim, rope_dim)``.

    ``c`` is the number of interleaved projection halves: 2 for ratio 4 and 1
    for ratio 128.  Raises :class:`TreeContractError` for unsupported values
    (D6).  See ``8947f1f9f:csrc/attention/compressor_tree_project/torch_binding.h:11-13``.
    """
    ratio = check_ratio(ratio)
    head_dim = check_head_dim(head_dim)
    rope_dim = check_rope_dim(rope_dim)
    c = 2 if ratio == 4 else 1
    return ratio, c, head_dim, rope_dim


# ---------------------------------------------------------------------------
# Device helpers (lazy; never imported at module import)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_num_vectorcores() -> int:
    """Return the Ascend vector-core count, or 32 when unavailable.

    The existing repo helper lives in
    ``sgl_kernel_npu/indexer/flash_block_score_decode.py`` but that module
    imports Triton at import time.  We therefore import it **inside** this
    function so importing ``tree._common`` never pulls in Triton.  On a host
    without Triton/NPU the function falls back to 32.
    """
    try:
        from ..indexer.flash_block_score_decode import _get_vectorcore_num_safe

        value = int(_get_vectorcore_num_safe())
        if value > 0:
            return value
    except Exception:
        pass

    try:
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            props = npu.get_device_properties(npu.current_device())
            value = int(getattr(props, "vector_core_num", 0) or 0)
            if value > 0:
                return value
    except Exception:
        pass

    return 32
