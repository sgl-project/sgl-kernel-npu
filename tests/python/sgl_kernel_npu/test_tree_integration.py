"""End-to-end DSV4 tree chain: project -> compressor -> indexer -> attention.

The four operators are independently tested in ``test_tree_*.py``.  This file
pins the contracts *between* them on one real tree geometry:

* ``compressor_tree_project`` writes the FP32 node rows consumed by
  ``compressor_tree``;
* ``compressor_tree`` produces the BF16 node KV pool consumed by attention as
  ``cmp_tree_kv``;
* ``quant_lightning_indexer_tree`` produces the C4 logical Top-K consumed by
  attention as ``cmp_sparse_indices``;
* ``sparse_attn_sharedkv_tree`` consumes both and returns the final output and
  LSE.

The oracle stays the independent CPU reference (D5).  The module skips cleanly
without ``torch_npu``/Triton, which is the CPU-only CI path.
"""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("triton")

if not torch.npu.is_available():
    pytest.skip("an Ascend NPU is required", allow_module_level=True)

from sgl_kernel_npu.tree import (  # noqa: E402
    compressor_tree,
    compressor_tree_project,
    quant_lightning_indexer_tree,
    sparse_attn_sharedkv_tree,
)
from tree_fixtures import (  # noqa: E402
    make_attention_case,
    make_compressor_case,
    make_indexer_case,
    make_project_case,
)
from tree_reference import (  # noqa: E402
    compressor_tree_project_ref,
    compressor_tree_ref,
    quant_lightning_indexer_tree_ref,
    sparse_attn_sharedkv_tree_ref,
)

DEVICE = torch.device("npu:0")
FP32_TOL = 2e-3
BF16_TOL = 1e-2
ATTN_TOL = 2e-2
LSE_TOL = 5e-2

# Same tree for every stage: one request, 17 verify rows.
_GEOMETRY = dict(batch=1, budget=16, depth=4, seed=777)
# Long enough that ``max(lengths) > topk``: this forces the indexer through
# the ranked score + Top-K merge path instead of the Python select-all shortcut.
_PREFIX = [2048]
_INDEXER_HEADS = 64
_INDEXER_HEAD_DIM = 128
_TOPK = 512


def _to_npu(value):
    if isinstance(value, torch.Tensor):
        return value.to(DEVICE)
    if isinstance(value, dict):
        return {k: _to_npu(v) for k, v in value.items()}
    return value


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def _indexer_prod_args(args):
    """Map the fixture's one-group tensors to the 64-head public contract."""
    query = args["query"]
    n, groups = int(query.shape[0]), int(query.shape[1])
    query_64 = torch.zeros(
        n, _INDEXER_HEADS, _INDEXER_HEAD_DIM, dtype=torch.int8
    )
    query_64[:, :groups] = query
    weights_64 = torch.zeros(n, _INDEXER_HEADS, dtype=torch.float16)
    weights_64[:, :groups] = args["weights"]
    scale_64 = torch.ones(n, _INDEXER_HEADS, dtype=torch.float16)
    scale_64[:, :groups] = args["qs"]
    return dict(
        query=query_64,
        history=args["history"],
        weights=weights_64,
        query_scale=scale_64,
        history_scale=args["history_scale"],
        nodes=args["tree_key"],
        node_scale=args["tree_scale"],
        ancestors=args["ancestors"],
        prefixes=args["prefixes"],
        requests=args["reqs"],
        table=args["block_table"],
        cu=args["cu"],
        lengths=args["lengths"],
        residual=args["residual"],
        metadata=args["metadata"],
        topk=int(args["topk"]),
    )


def _attention_args(case):
    return {k: v for k, v in case.items() if k not in ("spec", "cmp_mode")}


def test_project_compressor_indexer_attention_chain():
    project = make_project_case(
        seed=11, ratio=4, head_dim=512, hidden=1024, n=_GEOMETRY["budget"] + 1
    )
    project_npu = _to_npu(
        {
            k: project[k]
            for k in ("x", "wkv", "wgate", "nodes", "ape", "positions", "cu")
        }
    )
    # Native 7-arg ABI: ratio/head_dim are derived from APE.
    compressor_tree_project(
        project_npu["x"],
        project_npu["wkv"],
        project_npu["wgate"],
        project_npu["nodes"],
        project_npu["ape"],
        project_npu["positions"],
        project_npu["cu"],
    )
    torch.npu.synchronize()

    project_ref = make_project_case(
        seed=11, ratio=4, head_dim=512, hidden=1024, n=_GEOMETRY["budget"] + 1
    )
    compressor_tree_project_ref(
        project_ref["x"],
        project_ref["wkv"],
        project_ref["wgate"],
        project_ref["nodes"],
        project_ref["ape"],
        project_ref["positions"],
        project_ref["cu"],
        project_ref["ratio"],
        project_ref["head_dim"],
    )
    torch.testing.assert_close(
        _to_cpu(project_npu["nodes"]), project_ref["nodes"], rtol=FP32_TOL, atol=FP32_TOL
    )

    # ---- compressor_tree: consume the projected node pool -----------------
    comp = make_compressor_case(
        seed=_GEOMETRY["seed"],
        ratio=4,
        head_dim=512,
        rows=4,
        batch=_GEOMETRY["batch"],
        budget=_GEOMETRY["budget"],
        depth=_GEOMETRY["depth"],
        cos_pad=8,
        page_pad=8,
        prefixes=_PREFIX,
    )
    n = int(comp["n"])
    assert n == _GEOMETRY["budget"] + 1

    comp_args = dict(comp["args"])
    comp_args["nodes"] = _to_cpu(project_npu["nodes"])
    expected_nodes_kv = compressor_tree_ref(**comp_args)

    comp_npu = _to_npu(dict(comp["args"]))
    # Real node_pool.py allocates nodes as [n+1, 1, 1, 2*c*d].
    comp_npu["nodes"] = project_npu["nodes"].view(n + 1, 1, 1, -1)
    comp_npu["epsilon"] = comp_npu.pop("eps")
    actual_nodes_kv = compressor_tree(**comp_npu)
    torch.npu.synchronize()
    torch.testing.assert_close(
        _to_cpu(actual_nodes_kv),
        expected_nodes_kv,
        rtol=BF16_TOL,
        atol=BF16_TOL,
    )

    # ---- quant_lightning_indexer_tree: produce the C4 Top-K ----------------
    indexer = make_indexer_case(
        seed=_GEOMETRY["seed"],
        rows=4,
        topk=_TOPK,
        batch=_GEOMETRY["batch"],
        budget=_GEOMETRY["budget"],
        depth=_GEOMETRY["depth"],
        prefixes=_PREFIX,
    )
    assert int(indexer["lengths"].max()) > _TOPK, (
        "integration must exercise the ranked indexer path, not select-all"
    )
    # The oracle takes the fixture's native names/one-group shape; the
    # production op takes the 64-head public contract below.
    expected_topk = quant_lightning_indexer_tree_ref(**dict(indexer["args"]))
    indexer_args = _indexer_prod_args(indexer["args"])
    indexer_npu = _to_npu(indexer_args)
    actual_topk = quant_lightning_indexer_tree(
        indexer_npu["query"],
        indexer_npu["history"],
        indexer_npu["weights"],
        indexer_npu["query_scale"],
        indexer_npu["history_scale"],
        indexer_npu["nodes"],
        indexer_npu["node_scale"],
        indexer_npu["ancestors"],
        indexer_npu["prefixes"],
        indexer_npu["requests"],
        indexer_npu["table"],
        indexer_npu["cu"],
        indexer_npu["lengths"],
        indexer_npu["residual"],
        indexer_npu["metadata"],
        indexer_npu["topk"],
    )
    torch.npu.synchronize()
    assert actual_topk.shape == (n, 1, _TOPK)
    for row in range(n):
        assert sorted(actual_topk[row, 0].tolist()) == sorted(
            expected_topk[row, 0].tolist()
        )

    # ---- attention: consume the compressor node pool + indexer Top-K ------
    attention = make_attention_case(
        seed=_GEOMETRY["seed"],
        ratio=4,
        batch=_GEOMETRY["batch"],
        budget=_GEOMETRY["budget"],
        depth=_GEOMETRY["depth"],
        rows=4,
        window=127,
        sink=True,
        cmp_mode="c4",
        dtype=torch.bfloat16,
        heads=8,
        prefixes=_PREFIX,
    )
    attn_args_cpu = _attention_args(attention)
    attn_args_cpu["cmp_tree_kv"] = _to_cpu(actual_nodes_kv)
    attn_args_cpu["cmp_sparse_indices"] = _to_cpu(actual_topk)
    expected_out, expected_lse = sparse_attn_sharedkv_tree_ref(
        **attn_args_cpu, return_softmax_lse=True
    )

    attn_npu = _to_npu(_attention_args(attention))
    attn_npu["cmp_tree_kv"] = actual_nodes_kv
    attn_npu["cmp_sparse_indices"] = actual_topk
    q = attn_npu.pop("q")
    actual_out, actual_lse = sparse_attn_sharedkv_tree(
        q, return_softmax_lse=True, **attn_npu
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        _to_cpu(actual_out).float(),
        expected_out.float(),
        rtol=ATTN_TOL,
        atol=ATTN_TOL,
    )
    torch.testing.assert_close(
        _to_cpu(actual_lse).float(),
        expected_lse.float(),
        rtol=LSE_TOL,
        atol=LSE_TOL,
    )


def test_lazy_package_surface_exports_all_four_ops():
    import sgl_kernel_npu.tree as tree

    for name in (
        "compressor_tree_project",
        "compressor_tree",
        "quant_lightning_indexer_tree",
        "sparse_attn_sharedkv_tree",
    ):
        assert callable(getattr(tree, name))
    assert callable(tree.tree_project)
    assert callable(tree.tree_attention)
