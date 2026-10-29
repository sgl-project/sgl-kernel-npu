#!/usr/bin/env python3
"""Compare sgl-kernel-npu Triton tree ops with vllm-ascend _C_ascend ops.

Both implementations receive the *same* tensors and (for indexer/attention)
the *same* vendor metadata.  The script does not use the Python reference as a
performance baseline; it compares the two real implementations directly.

Prerequisites on the A3 host:
  * ``sgl_kernel_npu`` built/installed (new implementation)
  * vllm-ascend tree binding loaded, or ``SGLANG_DSPARK_TREE_OPS`` pointing at
    the built binding .so files
  * ``torch_npu`` and an Ascend device

Examples:
  python benchmark/spec_decode/compare_tree_ops.py --op all --warmup 1 --reps 3
  python benchmark/spec_decode/compare_tree_ops.py --op indexer --warmup 1 --reps 3 --json /tmp/cmp.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
TEST_DIR = ROOT / "tests" / "python" / "sgl_kernel_npu"
for path in (str(BENCH_DIR), str(TEST_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import bench_tree_ops as bench  # noqa: E402


def _load_vendor() -> Any:
    """Load the vllm-ascend ``torch.ops._C_ascend`` tree namespace."""
    try:
        import vllm_ascend.vllm_ascend_C  # noqa: F401
    except Exception:
        pass

    # Load the SGLang-side binding set as well.  vllm-ascend's import may
    # register the four tree ops but not the separate metadata bindings.
    try:
        from sglang.srt.speculative.dspark_components.tree.torch_ops import (
            load_tree_ops,
        )

        load_tree_ops()
    except Exception:
        pass
    raw = os.environ.get("SGLANG_DSPARK_TREE_OPS", "")
    for raw_path in raw.split(os.pathsep):
        if raw_path and Path(raw_path).is_file():
            try:
                torch.ops.load_library(raw_path)
            except Exception:
                pass

    ns = getattr(torch.ops, "_C_ascend", None)
    required = (
        "compressor_tree_project",
        "compressor_tree",
        "quant_lightning_indexer_tree",
        "npu_sparse_attn_sharedkv_tree",
    )
    missing = [name for name in required if not hasattr(ns, name)]
    if ns is None or missing:
        raise RuntimeError(
            "vllm-ascend _C_ascend tree ops are unavailable; missing "
            f"{missing}. Build/load vllm-ascend tree bindings first, or set "
            "SGLANG_DSPARK_TREE_OPS to the binding .so paths."
        )
    return ns


def _find_meta_op(*names: str) -> Any:
    """Find a metadata op registered by vllm-ascend or SGLang bindings."""
    for ns_name in ("custom", "_C_ascend", "npu", "cann_ops_transformer"):
        ns = getattr(torch.ops, ns_name, None)
        if ns is None:
            continue
        for name in names:
            op = getattr(ns, name, None)
            if op is not None:
                return op
    return None


def _build_metadata(ns: Any, op: str, kwargs: dict[str, Any]) -> torch.Tensor:
    """Build the vendor metadata for indexer/attention from the same inputs."""
    if op == "indexer":
        q = kwargs["query"]
        meta_op = _find_meta_op(
            "npu_quant_lightning_indexer_v2_metadata",
            "quant_lightning_indexer_v2_metadata",
            "quant_lightning_indexer_metadata",
        )
        if meta_op is None:
            raise RuntimeError(
                "no lightning-indexer metadata op found in torch.ops.custom/"
                "_C_ascend/npu/cann_ops_transformer; load the SGLang tree binding .so set"
            )
        try:
            return meta_op(
            num_heads_q=int(q.shape[1]),
            num_heads_k=1,
            head_dim=int(q.shape[2]),
            topk=int(kwargs["topk"]),
            quant_mode=2,
            cu_seqlens_q=kwargs["cu"].to(torch.int32),
            seqused_k=kwargs["lengths"].to(torch.int32),
            cmp_residual_k=kwargs["residual"].to(torch.int32),
            batch_size=int(q.shape[0]),
            max_seqlen_q=1,
            max_seqlen_k=max(1, int(kwargs["lengths"].max().item())),
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
                device=str(q.device),
            )
        except TypeError:
            # cann_ops_transformer's lower-level metadata op uses the
            # documented cu_seqlens_k/seqused_k spelling.
            return meta_op(
                cu_seqlens_q=kwargs["cu"].to(torch.int32),
                cu_seqlens_k=None,
                seqused_q=None,
                seqused_k=kwargs["lengths"].to(torch.int32),
                cmp_residual_k=kwargs["residual"].to(torch.int32),
                batch_size=int(q.shape[0]),
                max_seqlen_q=1,
                max_seqlen_k=max(1, int(kwargs["lengths"].max().item())),
                num_heads_q=int(q.shape[1]),
                num_heads_k=1,
                head_dim=int(q.shape[2]),
                topk=int(kwargs["topk"]),
                quant_mode=2,
                mask_mode=3,
                layout_q="TND",
                layout_k="PA_BBND",
                cmp_ratio=4,
            )

    if op == "attention":
        q = kwargs["q"]
        seq = kwargs["seqused_kv"].to(torch.int32)
        cu_ori = torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32)))
        cu_cmp = torch.cat(
            (
                seq.new_zeros(1),
                (seq // int(kwargs["cmp_ratio"])).cumsum(0, dtype=torch.int32),
            )
        )
        cmp_topk = (
            int(kwargs["cmp_sparse_indices"].shape[-1])
            if kwargs.get("cmp_sparse_indices") is not None
            and int(kwargs["cmp_ratio"]) == 4
            else 0
        )
        kwargs["cu_seqlens_ori_kv"] = cu_ori
        kwargs["cu_seqlens_cmp_kv"] = cu_cmp
        op = _find_meta_op(
            "npu_sparse_attn_sharedkv_metadata",
            "sparse_attn_sharedkv_metadata",
        )
        if op is None:
            raise RuntimeError(
                "no sparse-attn metadata op found in torch.ops.custom/"
                "_C_ascend/npu; load the SGLang tree binding .so set"
            )
        return op(
            num_heads_q=int(q.shape[1]),
            num_heads_kv=1,
            head_dim=int(q.shape[2]),
            seqused_q=q.new_empty((0,), dtype=torch.int32),
            batch_size=int(q.shape[0]),
            max_seqlen_q=1,
            max_seqlen_kv=max(1, int(seq.max().item())),
            cmp_topk=cmp_topk,
            has_ori_kv=True,
            has_cmp_kv=kwargs.get("cmp_kv") is not None,
            device=str(q.device),
            cu_seqlens_q=kwargs["cu_seqlens_q"].to(torch.int32),
            cu_seqlens_ori_kv=cu_ori,
            cu_seqlens_cmp_kv=cu_cmp,
            seqused_kv=seq,
            cmp_ratio=int(kwargs["cmp_ratio"]),
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=int(kwargs.get("ori_win_left", 127)),
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )

    raise ValueError(f"no metadata for {op}")


def _new_fn(op: str) -> Callable[..., Any]:
    import sgl_kernel_npu.tree as tree

    return getattr(tree, bench.OP_TO_PUBLIC[op])


def _vendor_call(ns: Any, op: str, kwargs: dict[str, Any]) -> Any:
    if op == "project":
        ns.compressor_tree_project(
            kwargs["x"],
            kwargs["wkv"],
            kwargs["wgate"],
            kwargs["nodes"],
            kwargs["ape"],
            kwargs["positions"],
            kwargs["cu"],
        )
        return kwargs["nodes"]
    if op == "compressor":
        return ns.compressor_tree(
            kwargs["state"],
            kwargs["nodes"],
            kwargs["norm"],
            kwargs["cos"],
            kwargs["sin"],
            kwargs["table"],
            kwargs["prefix"],
            kwargs["ancestors"],
            kwargs["depths"],
            kwargs["requests"],
            kwargs["ratio"],
            kwargs["rope_dim"],
            kwargs["epsilon"],
        )
    if op == "indexer":
        return ns.quant_lightning_indexer_tree(
            kwargs["query"],
            kwargs["history"],
            kwargs["weights"],
            kwargs["query_scale"],
            kwargs["history_scale"],
            kwargs["nodes"],
            kwargs["node_scale"],
            kwargs["ancestors"],
            kwargs["prefixes"],
            kwargs["requests"],
            kwargs["table"],
            kwargs["cu"],
            kwargs["lengths"],
            kwargs["residual"],
            kwargs["metadata"],
            kwargs["topk"],
        )
    if op == "attention":
        q = kwargs.pop("q")
        return ns.npu_sparse_attn_sharedkv_tree(q, **kwargs)
    raise ValueError(f"unknown op {op}")


def _new_call(op: str, fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    if op == "project":
        bench._invoke(fn, kwargs)
        return kwargs["nodes"]
    return bench._invoke(fn, kwargs)


def _compare(op: str, a: Any, b: Any) -> tuple[bool, str]:
    try:
        if op == "project":
            if a.shape != b.shape:
                return False, f"shape {a.shape} vs {b.shape}"
            if not torch.equal(a[0], b[0]):
                return False, "reserved row 0 differs"
            if not torch.allclose(a[1:], b[1:], rtol=2e-3, atol=2e-3):
                return False, "node rows differ"
            return True, "ok"
        if op == "compressor":
            if a.shape != b.shape:
                return False, f"shape {a.shape} vs {b.shape}"
            if not torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-2):
                return False, "compressor differs"
            return True, "ok"
        if op == "indexer":
            if a.shape != b.shape:
                return False, f"shape {a.shape} vs {b.shape}"
            if not torch.equal(torch.sort(a, dim=-1).values, torch.sort(b, dim=-1).values):
                return False, "top-k multiset differs"
            return True, "ok"
        if op == "attention":
            ao, al = a if isinstance(a, tuple) else (a, None)
            bo, bl = b if isinstance(b, tuple) else (b, None)
            if ao.shape != bo.shape:
                return False, f"shape {ao.shape} vs {bo.shape}"
            if not torch.allclose(ao.float(), bo.float(), rtol=2e-2, atol=2e-2):
                return False, "attention output differs"
            if al is not None and bl is not None and not torch.allclose(
                al.float(), bl.float(), rtol=5e-2, atol=5e-2
            ):
                return False, "attention LSE differs"
            return True, "ok"
    except Exception as exc:
        return False, f"compare error: {exc}"
    return False, f"unknown op {op}"


def main(argv: list[str] | None = None) -> int:
    args = bench._parse_args(argv)
    bench._require_npu(args.device)
    ns = _load_vendor()
    import sgl_kernel_npu.tree  # noqa: F401  (fail early if package is missing)

    ops = list(bench.OP_ORDER) if args.op == "all" else [args.op]
    results: list[dict[str, Any]] = []
    for op in ops:
        kwargs_cpu, meta = bench._build_case(op, args)
        kwargs = bench._move_case(kwargs_cpu, args.device)
        metadata_error: str | None = None
        if op in ("indexer", "attention"):
            try:
                kwargs["metadata"] = _build_metadata(ns, op, kwargs)
            except Exception as exc:
                metadata_error = f"{type(exc).__name__}: {exc}"
        if op == "indexer":
            # Keep the same names for both implementations; extra aliases are
            # filtered by signature-aware invocation.
            kwargs.setdefault("nodes", kwargs["tree_key"])
            kwargs.setdefault("node_scale", kwargs["tree_scale"])
            kwargs.setdefault("requests", kwargs["reqs"])
            kwargs.setdefault("table", kwargs["block_table"])

        new_fn = _new_fn(op)
        new_kwargs = bench._clone_case(kwargs)
        new_out = _new_call(op, new_fn, new_kwargs)
        new_times, _, _ = bench._time_npu(
            lambda: _new_call(op, new_fn, bench._clone_case(kwargs)),
            args.warmup,
            args.reps,
        )
        np50, np10, np90 = bench._percentiles(new_times)

        if metadata_error is not None:
            row = {
                "op": op,
                "new_p50_ms": np50,
                "new_p10_ms": np10,
                "new_p90_ms": np90,
                "vendor_p50_ms": None,
                "vendor_p10_ms": None,
                "vendor_p90_ms": None,
                "vendor_over_new": None,
                "correct": None,
                "detail": f"vendor skipped: {metadata_error}",
                "meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
            }
            results.append(row)
            print(f"{op:<12} new={np50!s:>10} ms  vendor=SKIP  correct=None  {metadata_error}")
            continue

        vendor_kwargs = bench._clone_case(kwargs)
        vendor_out = _vendor_call(ns, op, vendor_kwargs)
        torch.npu.synchronize()

        ok, detail = _compare(op, new_out, vendor_out)
        vendor_times, _, _ = bench._time_npu(
            lambda: _vendor_call(ns, op, bench._clone_case(kwargs)),
            args.warmup,
            args.reps,
        )
        vp50, vp10, vp90 = bench._percentiles(vendor_times)
        speedup = (
            vp50 / np50
            if vp50 is not None and np50 not in (None, 0)
            else None
        )
        row = {
            "op": op,
            "new_p50_ms": np50,
            "new_p10_ms": np10,
            "new_p90_ms": np90,
            "vendor_p50_ms": vp50,
            "vendor_p10_ms": vp10,
            "vendor_p90_ms": vp90,
            "vendor_over_new": speedup,
            "correct": ok,
            "detail": detail,
            "meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
        }
        results.append(row)
        print(
            f"{op:<12} new={np50!s:>10} ms  vendor={vp50!s:>10} ms  "
            f"vendor/new={speedup!s:>8}  correct={ok}  {detail}"
        )

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(f"[compare] wrote JSON: {args.json_path}")
    if any(r["correct"] is None for r in results):
        print("[compare] some vendor comparisons were skipped (see detail)")
    return 0 if all(r["correct"] is not False for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
