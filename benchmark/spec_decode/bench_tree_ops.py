#!/usr/bin/env python3
"""Benchmark / dry-run harness for the DSV4 DSpark tree operators.

Dry-run (CPU-only host: no Triton, no NPU, no ``libsgl_kernel_npu.so``):

    python benchmark/spec_decode/bench_tree_ops.py --op all --dry-run \
        --json /tmp/tree_dry.json --csv /tmp/tree_dry.csv

Dry-run imports the independent pure-torch oracle from
``tests/python/sgl_kernel_npu/tree_reference.py`` via an explicit ``sys.path``
bootstrap, generates deterministic inputs, runs the oracle, checks shape/dtype/
finiteness/determinism invariants, and writes schema-valid JSON/CSV.  It does
**not** import any production Triton module and does not fabricate device
measurements.  The latency columns in dry-run are real host wall-clock timings
of the CPU reference; ``speedup`` is left empty because it is not meaningful.

NPU (A3):

    python benchmark/spec_decode/bench_tree_ops.py --op all --baseline torch \
        --warmup 10 --reps 50 --csv perf/tree_perf.csv --json perf/tree_perf.json

Non-dry-run lazily imports ``torch_npu`` and the production tree kernels, times
with ``torch.npu.Event`` after ``torch.npu.synchronize()``, and raises a clear
error when no NPU is available.

The tree operators are defined in ``docs/tree_ops_semantics.md``.  The reference
functions are ``*_ref`` (D5) and are never imported by production code.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests" / "python" / "sgl_kernel_npu"

CSV_FIELDS = [
    "op",
    "case",
    "ratio",
    "head_dim",
    "hidden",
    "n",
    "batch",
    "topk",
    "dtype",
    "pages",
    "page_pad",
    "impl",
    "latency_ms_p10",
    "latency_ms_p50",
    "latency_ms_p90",
    "throughput",
    "speedup",
    "mem_mb",
    "correct",
    "ts",
]

OP_ORDER = ("project", "compressor", "indexer", "attention")
OP_TO_PUBLIC = {
    "project": "compressor_tree_project",
    "compressor": "compressor_tree",
    "indexer": "quant_lightning_indexer_tree",
    "attention": "sparse_attn_sharedkv_tree",
}
REFERENCE_NAMES = {
    "project": (
        "compressor_tree_project_ref",
        "tree_project_ref",
        "compressor_tree_project",
    ),
    "compressor": ("compressor_tree_ref", "tree_compressor_ref", "compressor_tree"),
    "indexer": (
        "quant_lightning_indexer_tree_ref",
        "tree_indexer_ref",
        "quant_lightning_indexer_tree",
    ),
    "attention": (
        "sparse_attn_sharedkv_tree_ref",
        "tree_attention_ref",
        "sparse_attn_sharedkv_tree_ref",
        "sparse_attn_sharedkv_tree",
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark/dry-run harness for DSV4 DSpark tree operators."
    )
    p.add_argument("--op", choices=(*OP_ORDER, "all"), default="all")
    p.add_argument("--ratio", type=int, default=4, choices=(4, 128))
    p.add_argument("--head-dim", type=int, default=512, choices=(128, 512))
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--nodes", type=int, default=17)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument(
        "--dtype",
        choices=("bf16", "bfloat16", "fp16", "float16"),
        default="bf16",
    )
    p.add_argument("--pages", type=int, default=4)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--page-pad", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--baseline", choices=("torch", "native", "both"), default="torch")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--csv", default=None)
    p.add_argument("--json", dest="json_path", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-on-mismatch", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)
    if args.nodes < 1:
        p.error("--nodes must be >= 1")
    if args.batch < 1:
        p.error("--batch must be >= 1")
    if args.depth < 1:
        p.error("--depth must be >= 1")
    if args.pages < 1:
        p.error("--pages must be >= 1")
    if args.block < 1:
        p.error("--block must be >= 1")
    if args.page_pad < 0:
        p.error("--page-pad must be >= 0")
    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.reps < 1:
        p.error("--reps must be >= 1")
    if not 1 <= args.topk <= 2048:
        p.error("--topk must be in [1, 2048] (tree indexer contract)")
    if args.hidden < 1:
        p.error("--hidden must be >= 1")
    if args.budget is not None and args.budget < 1:
        p.error("--budget must be >= 1")
    if args.budget is not None and args.budget < args.depth:
        p.error("--budget must be >= --depth for a tree")
    return args


def _torch_dtype(text: str) -> torch.dtype:
    text = text.lower()
    if text in ("bf16", "bfloat16"):
        return torch.bfloat16
    if text in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"unsupported dtype {text!r}")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype).replace("torch.", "")


# ---------------------------------------------------------------------------
# Reference loading (CPU-safe, explicit sys.path bootstrap)
# ---------------------------------------------------------------------------

def _bootstrap_reference_path() -> Path:
    if not TESTS_DIR.is_dir():
        raise RuntimeError(f"reference directory not found: {TESTS_DIR}")
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    importlib.invalidate_caches()
    return TESTS_DIR / "tree_reference.py"


def _load_reference(wait_seconds: float = 0.0) -> Any:
    """Import ``tree_reference`` from the tests directory.

    ``wait_seconds`` supports the Wave-1 race where another agent is still
    writing the reference.  Production code never imports this module.
    """
    path = _bootstrap_reference_path()
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    last_error: Exception | None = None
    while True:
        try:
            if "tree_reference" in sys.modules:
                return sys.modules["tree_reference"]
            return importlib.import_module("tree_reference")
        except Exception as exc:  # ImportError, SyntaxError, ...
            last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
            importlib.invalidate_caches()
            sys.modules.pop("tree_reference", None)
    raise RuntimeError(
        f"cannot import the independent reference {path}. "
        f"Last error: {last_error!r}. This is a Wave-1 blocker when the "
        "reference agent has not written the file yet; the harness does not "
        "substitute a fake oracle."
    ) from last_error


def _resolve_reference_fn(reference: Any, op: str) -> Callable[..., Any]:
    for name in REFERENCE_NAMES[op]:
        fn = getattr(reference, name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"tree_reference has no callable for {op!r} "
        f"(tried {REFERENCE_NAMES[op]})"
    )


# ---------------------------------------------------------------------------
# Signature-aware invocation
# ---------------------------------------------------------------------------

def _invoke(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Call ``fn`` with the subset of ``kwargs`` its signature accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    params = sig.parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    call_kwargs = dict(kwargs) if has_var_kw else {
        k: v for k, v in kwargs.items() if k in params
    }

    missing = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and name not in call_kwargs
    ]
    if missing:
        raise TypeError(f"{fn!r}: missing required arguments {missing}")

    pos_only = [
        name for name, p in params.items() if p.kind == p.POSITIONAL_ONLY
    ]
    if pos_only:
        args = [call_kwargs.pop(name) for name in pos_only]
        return fn(*args, **call_kwargs)
    return fn(**call_kwargs)


# ---------------------------------------------------------------------------
# Deterministic input builders
# ---------------------------------------------------------------------------

def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _randn(shape: Sequence[int], gen: torch.Generator, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = torch.randn(*shape, generator=gen, dtype=torch.float32)
    return t.to(dtype) if dtype is not None else t


def _randint(
    low: int,
    high: int,
    shape: Sequence[int],
    gen: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randint(low, high, shape, generator=gen, dtype=dtype)


def _make_tree(
    n: int,
    batch: int,
    depth: int,
    ratio: int,
    base_prefix: int,
    budget: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, ancestor-closed tree geometry.

    The tree is a chain of ``min(n/batch - 1, depth)`` nodes per request, with
    any remaining nodes attached one level below the chain tail.  This keeps
    every ancestor path within ``depth`` and exercises branching.  ``budget``
    caps the number of non-root nodes per request when supplied.
    """
    n = max(1, int(n))
    batch = max(1, min(int(batch), n))
    query_width = max(1, n // batch)
    if budget is not None:
        query_width = min(query_width, max(1, int(budget) + 1))
    n_eff = query_width * batch
    max_depth = max(1, int(depth))
    chain_len = min(query_width - 1, max_depth)

    prefixes: list[int] = []
    depths: list[int] = []
    requests: list[int] = []
    paths: list[list[int]] = []

    for r in range(batch):
        prefix = int(base_prefix) + r * max(1, int(ratio))
        parents: list[int] = [0] * query_width
        for i in range(1, query_width):
            if i <= chain_len:
                parents[i] = i - 1
            else:
                parents[i] = max(0, chain_len - 1)

        for i in range(query_width):
            if i == 0:
                path = [0]
            else:
                chain = [i]
                cur = i
                while cur > 0:
                    cur = parents[cur]
                    chain.append(cur)
                path = list(reversed(chain))
            prefixes.append(prefix)
            depths.append(len(path) - 1)
            requests.append(r)
            paths.append(path)

    width = max(depths) + 1
    ancestors = torch.full((n_eff, width), -1, dtype=torch.int64)
    for row, path in enumerate(paths):
        ancestors[row, : len(path)] = torch.tensor(path, dtype=torch.int64)

    prefix_t = torch.tensor(prefixes, dtype=torch.int32)
    depth_t = torch.tensor(depths, dtype=torch.int64)
    request_t = torch.tensor(requests, dtype=torch.int64)
    positions = ((prefix_t.long() + depth_t) % int(ratio)).to(torch.int32)
    seqused = (prefix_t.long() + depth_t + 1).to(torch.int32)
    cu = torch.arange(n_eff + 1, dtype=torch.int32)
    return {
        "n": n_eff,
        "batch": batch,
        "query_width": query_width,
        "prefix": prefix_t,
        "depths": depth_t,
        "requests": request_t,
        "ancestors": ancestors,
        "positions": positions,
        "seqused": seqused,
        "cu": cu,
        "max_depth": width - 1,
    }


def _ceil_div(a: int, b: int) -> int:
    return -((-int(a)) // int(b))


def _align_up(value: int, alignment: int = 128) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _make_page_table(
    batch: int,
    max_pos: int,
    block: int,
    pages: int,
    gen: torch.Generator,
) -> torch.Tensor:
    width = max(1, _ceil_div(max(0, int(max_pos)), int(block)) + 1)
    table = torch.empty((int(batch), width), dtype=torch.int32)
    for r in range(int(batch)):
        ids = (torch.arange(width, dtype=torch.int32) + r) % int(pages)
        table[r] = ids
    return table


def _build_project(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    gen = _gen(args.seed)
    ratio = int(args.ratio)
    c = 2 if ratio == 4 else 1
    d = int(args.head_dim)
    hidden = int(args.hidden)
    geom = _make_tree(args.nodes, args.batch, args.depth, ratio, 127 if ratio == 128 else 3, args.budget)
    n = geom["n"]

    # project is BF16-only at the binding boundary.
    x = _randn((n, hidden), gen, torch.bfloat16)
    wkv = (_randn((c * d, hidden), gen) * 0.02).to(torch.bfloat16)
    wgate = (_randn((c * d, hidden), gen) * 0.02).to(torch.bfloat16)
    nodes = (_randn((n + 1, 1, 2 * c * d), gen) * 0.05).to(torch.float32)
    sentinel = torch.linspace(1.0, 2.0, nodes[0].numel(), dtype=torch.float32).reshape(nodes[0].shape)
    nodes[0].copy_(sentinel)
    ape = (_randn((ratio, c * d), gen) * 0.01).to(torch.float32)

    kwargs = {
        "x": x,
        "wkv": wkv,
        "wgate": wgate,
        "nodes": nodes,
        "ape": ape,
        "positions": geom["positions"],
        "cu": geom["cu"],
        "ratio": ratio,
        "head_dim": d,
    }
    meta = {
        "ratio": ratio,
        "head_dim": d,
        "hidden": hidden,
        "n": n,
        "batch": geom["batch"],
        "topk": int(args.topk),
        "dtype": "bf16",
        "pages": int(args.pages),
        "page_pad": int(args.page_pad),
        "sentinel": sentinel.clone(),
        "c": c,
        "d": d,
        "max_prefix": int(geom["prefix"].max().item()),
        "max_depth": int(geom["max_depth"]),
        "max_seqused": int(geom["seqused"].max().item()),
    }
    return kwargs, meta


def _build_compressor(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    gen = _gen(args.seed + 1)
    ratio = int(args.ratio)
    c = 2 if ratio == 4 else 1
    d = int(args.head_dim)
    rope = 64
    pad = int(args.page_pad)
    pages = int(args.pages)
    block = int(args.block)
    geom = _make_tree(args.nodes, args.batch, args.depth, ratio, 127 if ratio == 128 else 3, args.budget)
    n = geom["n"]
    batch = geom["batch"]

    if pad:
        wide = _randn((pages, block, 2 * c * d + pad), gen).to(torch.float32)
        state = wide[:, :, : 2 * c * d]
    else:
        state = _randn((pages, block, 2 * c * d), gen).to(torch.float32)

    nodes = (_randn((n + 1, 1, 2 * c * d), gen) * 0.05).to(torch.float32)
    nodes[0].zero_()
    norm = (_randn((d,), gen).abs() + 0.5).to(torch.float32)

    cos_rows = int(geom["prefix"].max().item()) + int(geom["max_depth"]) + ratio + 2
    cos_storage = _randn((cos_rows, rope + pad), gen).to(torch.float32)
    sin_storage = _randn((cos_rows, rope + pad), gen).to(torch.float32)
    cos = cos_storage[:, :rope]  # stride(1) == 1, row stride may be padded
    sin = sin_storage[:, :rope]

    max_prefix = int(geom["prefix"].max().item())
    table = _make_page_table(batch, max_prefix, block, pages, gen)
    eps = 1e-6

    kwargs = {
        "state": state,
        "nodes": nodes,
        "norm": norm,
        "cos": cos,
        "sin": sin,
        "table": table,
        "prefix": geom["prefix"],
        "ancestors": geom["ancestors"],
        "depths": geom["depths"],
        "requests": geom["requests"],
        "ratio": ratio,
        "rope_dim": rope,
        "eps": eps,
        # aliases accepted by some reference revisions:
        "rope": rope,
        "epsilon": eps,
    }
    meta = {
        "ratio": ratio,
        "head_dim": d,
        "hidden": int(args.hidden),
        "n": n,
        "batch": batch,
        "topk": int(args.topk),
        "dtype": "bf16",
        "pages": pages,
        "page_pad": pad,
        "c": c,
        "d": d,
        "max_prefix": max_prefix,
        "max_depth": int(geom["max_depth"]),
        "max_seqused": int(geom["seqused"].max().item()),
    }
    return kwargs, meta


def _build_indexer(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    gen = _gen(args.seed + 2)
    ratio = 4  # indexer is C4-only
    head_dim = 128
    pages = int(args.pages)
    block = int(args.block)
    pad = int(args.page_pad)
    topk = min(int(args.topk), 2048)
    # Force the scoring path (not the select-all shortcut): lengths.max() must
    # exceed topk, i.e. prefix/4 > topk.  This keeps the dry-run honest about
    # exercising the reference's scoring loop.
    base_prefix = max(8, 4 * (topk + 4))
    geom = _make_tree(args.nodes, args.batch, args.depth, ratio, base_prefix, args.budget)
    n = geom["n"]
    batch = geom["batch"]
    max_prefix = int(geom["prefix"].max().item())

    query = _randint(-128, 127, (n, 64, 128), gen, torch.int8)
    history_storage = _randint(-128, 127, (pages, block + pad, 1, 128), gen, torch.int8)
    history = history_storage[:, :block]
    hs_storage = (_randn((pages, block + pad, 1, 1), gen).abs() + 0.1).to(torch.float16)
    history_scale = hs_storage[:, :block]
    weights = (_randn((n, 64), gen) * 0.1).to(torch.float16)
    qs = (_randn((n, 64), gen).abs() + 0.1).to(torch.float16)
    tree_key = _randint(-128, 127, (n, 128), gen, torch.int8)
    tree_scale = (_randn((n,), gen).abs() + 0.1).to(torch.float16)
    table = _make_page_table(batch, max_prefix // ratio, block, pages, gen)
    seq = geom["seqused"].long()
    lengths = (seq // ratio).to(torch.int32)
    residual = (seq % ratio).to(torch.int32)

    kwargs = {
        "query": query,
        "history": history,
        "history_scale": history_scale,
        "weights": weights,
        "qs": qs,
        "query_scale": qs,
        "tree_key": tree_key,
        "tree_scale": tree_scale,
        # Production aliases: the public op uses nodes/node_scale/requests/table.
        "nodes": tree_key,
        "node_scale": tree_scale,
        "ancestors": geom["ancestors"],
        "prefixes": geom["prefix"],
        "reqs": geom["requests"],
        "requests": geom["requests"],
        "block_table": table,
        "table": table,
        "cu": geom["cu"],
        "lengths": lengths,
        "residual": residual,
        "metadata": torch.zeros(1024, dtype=torch.int32),
        "topk": topk,
        "head_dim": head_dim,
    }
    meta = {
        "ratio": ratio,
        "head_dim": head_dim,
        "hidden": int(args.hidden),
        "n": n,
        "batch": batch,
        "topk": topk,
        "dtype": "int8",
        "pages": pages,
        "page_pad": pad,
        "max_prefix": max_prefix,
        "max_depth": int(geom["max_depth"]),
        "max_seqused": int(geom["seqused"].max().item()),
    }
    return kwargs, meta


def _build_attention(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    gen = _gen(args.seed + 3)
    ratio = int(args.ratio)
    head_dim = 512
    n1 = 64
    dtype = _torch_dtype(args.dtype)
    pages = int(args.pages)
    block = int(args.block)
    pad = int(args.page_pad)
    base_prefix = 127 if ratio == 128 else 8
    geom = _make_tree(args.nodes, args.batch, args.depth, ratio, base_prefix, args.budget)
    t = geom["n"]
    batch = geom["batch"]
    max_prefix = int(geom["prefix"].max().item())
    seqused = geom["seqused"]
    min_seqused = int(seqused.min().item())

    q = _randn((t, n1, head_dim), gen, dtype)
    ori_storage = _randn((pages, block + pad, 1, head_dim), gen, dtype)
    ori_kv = ori_storage[:, :block]
    cmp_storage = _randn((pages, block + pad, 1, head_dim), gen, dtype)
    cmp_kv = cmp_storage[:, :block]

    width = _align_up(min(min_seqused, 128), 128)
    ori_sparse = torch.full((t, 1, width), -1, dtype=torch.int32)
    physical = torch.arange(width, dtype=torch.int32) % max(1, pages * block)
    for row in range(t):
        valid = min(width, int(seqused[row].item()))
        if valid:
            ori_sparse[row, 0, :valid] = physical[:valid]

    cmp_sparse: torch.Tensor | None
    if ratio == 4:
        # The C4 production contract uses K in {512, 1024}; use the smaller
        # width so the dry-run also exercises absent-ancestor masking.
        k = min(int(args.topk), 512)
        cmp_sparse = torch.arange(k, dtype=torch.int32).view(1, 1, k).expand(t, 1, k).contiguous()
    else:
        cmp_sparse = None

    # The reference indexes a block table by query row (T rows), while the
    # production op indexes by request.  One row per query is valid for both;
    # duplicate the request page map across its query rows.
    ori_table = _make_page_table(t, max_prefix, block, pages, gen)
    cmp_table = _make_page_table(t, max_prefix // ratio, block, pages, gen)
    cu_q = geom["cu"]
    cu_ori = torch.cat(
        (torch.zeros(1, dtype=torch.int32), seqused.long().cumsum(0).to(torch.int32))
    )
    cu_cmp = torch.cat(
        (torch.zeros(1, dtype=torch.int32), (seqused.long() // ratio).cumsum(0).to(torch.int32))
    )
    sinks = (_randn((n1,), gen) * 0.01).to(torch.float32)
    cmp_tree_kv = (_randn((t, head_dim), gen) * 0.1).to(dtype)

    kwargs = {
        "q": q,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "ori_sparse_indices": ori_sparse,
        "cmp_sparse_indices": cmp_sparse,
        "ori_block_table": ori_table,
        "cmp_block_table": cmp_table,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_ori_kv": cu_ori,
        "cu_seqlens_cmp_kv": cu_cmp,
        "seqused_q": torch.ones(t, dtype=torch.int32),
        "seqused_kv": seqused,
        "sinks": sinks,
        "metadata": torch.zeros(1024, dtype=torch.int32),
        "cmp_tree_kv": cmp_tree_kv,
        "tree_ancestors": geom["ancestors"],
        "tree_prefix": geom["prefix"],
        "softmax_scale": float(head_dim) ** -0.5,
        "cmp_ratio": ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_ND",
        "return_softmax_lse": True,
    }
    meta = {
        "ratio": ratio,
        "head_dim": head_dim,
        "hidden": int(args.hidden),
        "n": t,
        "batch": batch,
        "topk": int(args.topk),
        "dtype": _dtype_name(dtype),
        "pages": pages,
        "page_pad": pad,
        "max_prefix": max_prefix,
        "max_depth": int(geom["max_depth"]),
        "max_seqused": int(seqused.max().item()),
        "n1": n1,
    }
    return kwargs, meta


def _build_case(op: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if op == "project":
        return _build_project(args)
    if op == "compressor":
        return _build_compressor(args)
    if op == "indexer":
        return _build_indexer(args)
    if op == "attention":
        return _build_attention(args)
    raise ValueError(f"unknown op {op!r}")


# ---------------------------------------------------------------------------
# Invariants, comparison, and timing
# ---------------------------------------------------------------------------

def _percentiles(times_ms: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not times_ms:
        return None, None, None
    values = sorted(float(x) for x in times_ms)

    def at(p: float) -> float:
        idx = int(round(p * (len(values) - 1)))
        return values[max(0, min(len(values) - 1, idx))]

    return at(0.10), at(0.50), at(0.90)


def _check_invariants(op: str, result: Any, meta: dict[str, Any]) -> tuple[bool, str]:
    try:
        if op == "project":
            nodes = result
            n, c, d = int(meta["n"]), int(meta["c"]), int(meta["d"])
            if tuple(nodes.shape) != (n + 1, 1, 2 * c * d):
                return False, f"project shape {tuple(nodes.shape)} != {(n + 1, 1, 2 * c * d)}"
            if nodes.dtype != torch.float32:
                return False, f"project dtype {nodes.dtype} != float32"
            if not torch.equal(nodes[0].reshape(-1), meta["sentinel"].reshape(-1)):
                return False, "project modified reserved row 0"
            if not bool(torch.isfinite(nodes[1:]).all()):
                return False, "project produced non-finite node rows"
            if not bool((nodes[1:] != 0).any()):
                return False, "project did not write any node row"
            return True, "ok"

        if op == "compressor":
            out = result
            n, d = int(meta["n"]), int(meta["d"])
            if tuple(out.shape) != (n, d):
                return False, f"compressor shape {tuple(out.shape)} != {(n, d)}"
            if out.dtype != torch.bfloat16:
                return False, f"compressor dtype {out.dtype} != bfloat16"
            if not bool(torch.isfinite(out).all()):
                return False, "compressor produced non-finite values"
            if not bool((out != 0).any()):
                return False, "compressor produced only zero rows (no emitting node)"
            return True, "ok"

        if op == "indexer":
            out = result
            n, topk = int(meta["n"]), int(meta["topk"])
            if tuple(out.shape) != (n, 1, topk):
                return False, f"indexer shape {tuple(out.shape)} != {(n, 1, topk)}"
            if out.dtype != torch.int32:
                return False, f"indexer dtype {out.dtype} != int32"
            bound = int(meta["max_prefix"]) // 4 + n + 1
            if bool((out < -1).any()) or bool((out >= bound).any()):
                return False, f"indexer indices outside [-1, {bound})"
            if not bool((out >= 0).any()):
                return False, "indexer produced no valid indices"
            return True, "ok"

        if op == "attention":
            out, lse = result if isinstance(result, tuple) else (result, None)
            t, n1 = int(meta["n"]), int(meta["n1"])
            if tuple(out.shape) != (t, n1, 512):
                return False, f"attention out shape {tuple(out.shape)} != {(t, n1, 512)}"
            expected_dtype = _torch_dtype(meta["dtype"])
            if out.dtype != expected_dtype:
                return False, f"attention dtype {out.dtype} != {expected_dtype}"
            if not bool(torch.isfinite(out).all()):
                return False, "attention produced non-finite output"
            if not bool((out != 0).any()):
                return False, "attention produced an all-zero output"
            if lse is not None:
                if lse.numel() != t * n1:
                    return False, f"lse numel {lse.numel()} != {t * n1}"
                if not bool(torch.isfinite(lse).all()):
                    return False, "attention produced non-finite lse"
            return True, "ok"
    except Exception as exc:  # defensive: never let an invariant crash the run
        return False, f"invariant check failed: {exc}"
    return False, f"unknown op {op!r}"


def _same_result(op: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, tuple) or isinstance(b, tuple):
        if not (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)):
            return False
        return all(_same_result(op, x, y) for x, y in zip(a, b))
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return False
    if a.shape != b.shape:
        return False
    if op in ("indexer",):
        return bool(torch.equal(a, b))
    return bool(torch.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True))


def _time_cpu(
    call: Callable[[], Any],
    warmup: int,
    reps: int,
) -> tuple[list[float], Any, Any]:
    warmup = max(0, int(warmup))
    reps = max(1, int(reps))
    first: Any = None
    last: Any = None
    times: list[float] = []
    total = max(2, warmup + reps)
    for i in range(total):
        start = time.perf_counter()
        out = call()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if i == 0:
            first = out
        last = out
        if i >= warmup:
            times.append(elapsed_ms)
    return times, first, last


def _time_npu(
    call: Callable[[], Any],
    warmup: int,
    reps: int,
) -> tuple[list[float], Any, Any]:
    for _ in range(max(0, int(warmup))):
        call()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    first: Any = None
    last: Any = None
    times: list[float] = []
    for i in range(max(1, int(reps))):
        start.record()
        out = call()
        end.record()
        torch.npu.synchronize()
        times.append(float(start.elapsed_time(end)))
        if i == 0:
            first = out
        last = out
    return times, first, last


def _throughput(n: int, p50_ms: float | None) -> float | None:
    if p50_ms is None or p50_ms <= 0:
        return None
    return float(n) / (float(p50_ms) / 1000.0)


def _make_row(
    *,
    op: str,
    meta: dict[str, Any],
    impl: str,
    times: Sequence[float],
    speedup: float | None,
    mem_mb: float | None,
    correct: bool,
    note: str | None = None,
) -> dict[str, Any]:
    p10, p50, p90 = _percentiles(times)
    case = (
        f"r{meta['ratio']}-d{meta['head_dim']}-{meta['dtype']}-"
        f"n{meta['n']}-b{meta['batch']}-k{meta['topk']}-"
        f"p{meta['pages']}-pad{meta['page_pad']}"
    )
    row: dict[str, Any] = {
        "op": op,
        "case": case,
        "ratio": int(meta["ratio"]),
        "head_dim": int(meta["head_dim"]),
        "hidden": int(meta["hidden"]),
        "n": int(meta["n"]),
        "batch": int(meta["batch"]),
        "topk": int(meta["topk"]),
        "dtype": meta["dtype"],
        "pages": int(meta["pages"]),
        "page_pad": int(meta["page_pad"]),
        "impl": impl,
        "latency_ms_p10": p10,
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "throughput": _throughput(int(meta["n"]), p50),
        "speedup": speedup,
        "mem_mb": mem_mb,
        "correct": bool(correct),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if note:
        row["note"] = note
    return row


# ---------------------------------------------------------------------------
# Dry-run execution
# ---------------------------------------------------------------------------

def _dry_run_op(op: str, args: argparse.Namespace, reference: Any) -> list[dict[str, Any]]:
    kwargs, meta = _build_case(op, args)
    fn = _resolve_reference_fn(reference, op)

    if op == "project":
        def call() -> torch.Tensor:
            _invoke(fn, kwargs)
            return kwargs["nodes"].clone()
    else:
        def call() -> Any:
            return _invoke(fn, kwargs)

    times, first, last = _time_cpu(call, args.warmup, args.reps)
    ok, detail = _check_invariants(op, last, meta)
    deterministic = _same_result(op, first, last)
    correct = bool(ok and deterministic)
    note = (
        "dry-run: CPU reference; latency is host wall time; "
        f"invariants={'ok' if ok else detail}; deterministic={deterministic}"
    )
    return [
        _make_row(
            op=op,
            meta=meta,
            impl="reference-cpu",
            times=times,
            speedup=None,
            mem_mb=None,
            correct=correct,
            note=note,
        )
    ]


# ---------------------------------------------------------------------------
# NPU execution
# ---------------------------------------------------------------------------

def _require_npu(device: str) -> None:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "non-dry-run requires torch_npu. Use --dry-run on a CPU-only host."
        ) from exc
    if not torch.npu.is_available():
        raise RuntimeError(
            "non-dry-run requires an Ascend NPU; torch.npu.is_available() is False. "
            "Use --dry-run for the CPU reference."
        )
    torch.npu.set_device(device)


def _resolve_production_fn(op: str) -> Callable[..., Any]:
    import sgl_kernel_npu.tree as tree  # lazy: loads the built package on NPU only

    public = OP_TO_PUBLIC[op]
    fn = getattr(tree, public)
    if not callable(fn):
        raise ImportError(f"sgl_kernel_npu.tree.{public} is not callable")
    return fn


def _clone_case(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}


def _move_case(kwargs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }


def _make_call(op: str, fn: Callable[..., Any], kwargs: dict[str, Any]) -> Callable[[], Any]:
    if op == "project":
        def call() -> torch.Tensor:
            _invoke(fn, kwargs)
            return kwargs["nodes"].clone()
    else:
        def call() -> Any:
            return _invoke(fn, kwargs)
    return call


def _compare_outputs(op: str, prod: Any, ref: Any) -> tuple[bool, str]:
    try:
        if op == "project":
            if tuple(prod.shape) != tuple(ref.shape):
                return False, f"project shape mismatch {tuple(prod.shape)} vs {tuple(ref.shape)}"
            if not torch.equal(prod[0], ref[0]):
                return False, "project reserved row 0 differs"
            if not bool(torch.allclose(prod[1:], ref[1:], rtol=2e-3, atol=2e-3)):
                return False, "project node rows exceed tolerance"
            return True, "ok"
        if op == "compressor":
            if tuple(prod.shape) != tuple(ref.shape):
                return False, f"compressor shape mismatch {tuple(prod.shape)} vs {tuple(ref.shape)}"
            if not bool(torch.allclose(prod.float(), ref.float(), rtol=1e-2, atol=1e-2)):
                return False, "compressor output exceeds tolerance"
            return True, "ok"
        if op == "indexer":
            if tuple(prod.shape) != tuple(ref.shape):
                return False, f"indexer shape mismatch {tuple(prod.shape)} vs {tuple(ref.shape)}"
            prod_sorted = torch.sort(prod, dim=-1).values
            ref_sorted = torch.sort(ref, dim=-1).values
            if not bool(torch.equal(prod_sorted, ref_sorted)):
                return False, "indexer top-k multiset differs"
            return True, "ok"
        if op == "attention":
            prod_out, prod_lse = prod if isinstance(prod, tuple) else (prod, None)
            ref_out, ref_lse = ref if isinstance(ref, tuple) else (ref, None)
            if tuple(prod_out.shape) != tuple(ref_out.shape):
                return False, f"attention shape mismatch {tuple(prod_out.shape)} vs {tuple(ref_out.shape)}"
            if not bool(torch.allclose(prod_out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)):
                return False, "attention output exceeds tolerance"
            if prod_lse is not None and ref_lse is not None:
                if not bool(torch.allclose(prod_lse.float(), ref_lse.float(), rtol=5e-2, atol=5e-2)):
                    return False, "attention LSE exceeds tolerance"
            return True, "ok"
    except Exception as exc:
        return False, f"comparison failed: {exc}"
    return False, f"unknown op {op!r}"


def _native_baseline(op: str, kwargs: dict[str, Any]) -> tuple[Callable[[], Any] | None, str]:
    """Return a native baseline callable or a clear unavailability reason."""
    if op in ("project", "compressor"):
        return None, (
            "torch.ops.npu.compressor is the linear (non-tree) compressor and "
            "has no branch/ancestor semantics; no native tree baseline is mapped"
        )
    if op == "indexer":
        return None, "no registered native tree indexer baseline in torch.ops.npu"
    if op == "attention":
        if kwargs.get("cmp_kv") is not None:
            return None, (
                "torch.ops.npu.sparse_attn_sharedkv does not accept "
                "cmp_tree_kv/tree_ancestors/tree_prefix; the tree path needs "
                "the new production kernel (use --baseline torch)"
            )
        native = getattr(getattr(torch.ops, "npu", None), "sparse_attn_sharedkv", None)
        if native is None:
            return None, "torch.ops.npu.sparse_attn_sharedkv is not registered"
        return None, "native SWA mapping is not implemented in this harness"
    return None, f"no native baseline mapping for {op!r}"


def _npu_run_op(op: str, args: argparse.Namespace, reference: Any) -> list[dict[str, Any]]:
    _require_npu(args.device)
    kwargs_cpu, meta = _build_case(op, args)
    kwargs = _move_case(kwargs_cpu, args.device)
    ref_fn = _resolve_reference_fn(reference, op)
    prod_fn = _resolve_production_fn(op)

    # Production.
    prod_kwargs = _clone_case(kwargs)
    prod_call = _make_call(op, prod_fn, prod_kwargs)
    prod_times, prod_first, prod_last = _time_npu(prod_call, args.warmup, args.reps)

    # Torch reference on the same device.
    ref_kwargs = _clone_case(kwargs)
    ref_call = _make_call(op, ref_fn, ref_kwargs)
    ref_times, ref_first, ref_last = _time_npu(ref_call, args.warmup, args.reps)

    correct, detail = _compare_outputs(op, prod_last, ref_last)
    prod_p10, prod_p50, prod_p90 = _percentiles(prod_times)
    ref_p10, ref_p50, ref_p90 = _percentiles(ref_times)

    try:
        torch.npu.reset_peak_memory_stats()
        before = int(torch.npu.max_memory_allocated())
        prod_call()
        torch.npu.synchronize()
        after = int(torch.npu.max_memory_allocated())
        mem_mb: float | None = max(0, after - before) / (1024.0 * 1024.0)
    except Exception:
        mem_mb = None

    rows: list[dict[str, Any]] = [
        _make_row(
            op=op,
            meta=meta,
            impl="triton",
            times=prod_times,
            speedup=1.0,
            mem_mb=mem_mb,
            correct=correct,
            note=detail,
        )
    ]

    baselines: list[str]
    if args.baseline == "torch":
        baselines = ["torch"]
    elif args.baseline == "native":
        baselines = ["native"]
    else:
        baselines = ["torch", "native"]

    for baseline in baselines:
        if baseline == "torch":
            speedup = (
                (ref_p50 / prod_p50)
                if (ref_p50 is not None and prod_p50 not in (None, 0))
                else None
            )
            rows.append(
                _make_row(
                    op=op,
                    meta=meta,
                    impl="torch-reference",
                    times=ref_times,
                    speedup=speedup,
                    mem_mb=None,
                    correct=True,
                    note="torch reference on NPU; speedup = baseline_p50 / triton_p50",
                )
            )
            continue

        native_call, reason = _native_baseline(op, kwargs)
        if native_call is None:
            if args.baseline == "native":
                raise RuntimeError(f"native baseline unavailable for {op}: {reason}")
            print(f"[bench] {op}: native baseline skipped: {reason}", file=sys.stderr)
            continue
        native_times, native_first, native_last = _time_npu(
            native_call, args.warmup, args.reps
        )
        native_correct, native_detail = _compare_outputs(op, native_last, ref_last)
        native_p50 = _percentiles(native_times)[1]
        speedup = (
            (native_p50 / prod_p50)
            if (native_p50 is not None and prod_p50 not in (None, 0))
            else None
        )
        rows.append(
            _make_row(
                op=op,
                meta=meta,
                impl="native",
                times=native_times,
                speedup=speedup,
                mem_mb=None,
                correct=native_correct,
                note=native_detail,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace, ops_run: list[str]) -> None:
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dry_run": bool(args.dry_run),
            "device": "cpu" if args.dry_run else args.device,
            "ops_run": ops_run,
            "args": vars(args),
            "results": rows,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        print(f"[bench] wrote JSON: {path}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: ("" if row.get(k) is None else row.get(k))
                        for k in CSV_FIELDS
                    }
                )
        print(f"[bench] wrote CSV: {path}")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print(f"{'op':<12} {'impl':<16} {'p50_ms':>10} {'speedup':>9} {'correct':>8}")
    print("-" * 62)
    for row in rows:
        p50 = row.get("latency_ms_p50")
        p50_text = f"{p50:.4f}" if isinstance(p50, (int, float)) else "-"
        speed = row.get("speedup")
        speed_text = f"{speed:.3f}" if isinstance(speed, (int, float)) else "-"
        print(
            f"{row.get('op',''):<12} {row.get('impl',''):<16} "
            f"{p50_text:>10} {speed_text:>9} {str(bool(row.get('correct'))):>8}"
        )


def _error_row(op: str, args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    meta = {
        "ratio": int(args.ratio),
        "head_dim": 512 if op == "attention" else int(args.head_dim),
        "hidden": int(args.hidden),
        "n": int(args.nodes),
        "batch": int(args.batch),
        "topk": int(args.topk),
        "dtype": "int8" if op == "indexer" else args.dtype,
        "pages": int(args.pages),
        "page_pad": int(args.page_pad),
    }
    return _make_row(
        op=op,
        meta=meta,
        impl="error",
        times=[],
        speedup=None,
        mem_mb=None,
        correct=False,
        note=f"{type(exc).__name__}: {exc}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        args.device = "cpu"

    ops = list(OP_ORDER) if args.op == "all" else [args.op]
    rows: list[dict[str, Any]] = []
    had_error = False

    wait_s = float(os.environ.get("TREE_BENCH_REFERENCE_WAIT_S", "5" if args.dry_run else "0"))
    try:
        reference = _load_reference(wait_seconds=wait_s)
    except Exception as exc:
        print(f"[bench] ERROR: {exc}", file=sys.stderr)
        return 2

    for op in ops:
        try:
            if args.dry_run:
                op_rows = _dry_run_op(op, args, reference)
            else:
                op_rows = _npu_run_op(op, args, reference)
            rows.extend(op_rows)
        except Exception as exc:
            had_error = True
            print(f"[bench] {op} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            rows.append(_error_row(op, args, exc))

    _write_outputs(rows, args, ops)
    _print_summary(rows)

    mismatches = [row for row in rows if row.get("correct") is False]
    if had_error:
        print(f"[bench] {len(mismatches)} failing row(s); see notes above.", file=sys.stderr)
        return 2
    if mismatches and args.fail_on_mismatch:
        print(
            f"[bench] --fail-on-mismatch: {len(mismatches)} row(s) failed correctness.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


