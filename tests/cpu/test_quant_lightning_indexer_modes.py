"""CPU numerical and schema contracts for the A3 INT8 candidate extension."""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu"


@pytest.fixture
def modules():
    namespace = "_qli_cpu_test"
    packages = {}
    for suffix in (
        "",
        ".operators",
        ".operators.indexer",
        ".implementation",
        ".implementation.torch_native",
        ".implementation.torch_native.indexer",
    ):
        module = ModuleType(namespace + suffix)
        module.__path__ = (
            [str(PACKAGE.joinpath(*suffix.strip(".").split(".")))]
            if suffix
            else [str(PACKAGE)]
        )
        packages[module.__name__] = module
    before = set(sys.modules)
    with patch.dict(sys.modules, packages):
        try:
            reference = importlib.import_module(
                namespace
                + ".implementation.torch_native.indexer.quant_lightning_indexer"
            )
            validator = importlib.import_module(
                namespace + ".operators.indexer.indexer_utils"
            )
            yield (
                reference.quant_lightning_indexer,
                validator.validate_quant_indexer_candidates,
            )
        finally:
            for name in set(sys.modules) - before:
                if name.startswith(namespace):
                    sys.modules.pop(name, None)


def inputs(ratio=1):
    length = 16448  # More than 2048 blocks, so candidate pruning is exercised.
    pages = (length + 63) // 64
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).view(1, -1)
    q = torch.ones(1, 32, 128, dtype=torch.int8)
    k = torch.ones(pages, 64, 1, 128, dtype=torch.int8)
    ks = torch.zeros(pages, 64, 1, dtype=torch.float16)
    for position in range(length):
        scale = (
            (position + 1) / 1024
            if position < length - 8
            else 1001 + position - (length - 8)
        )
        ks[table[0, position // 64], position % 64, 0] = scale
    args = (
        q,
        k,
        torch.ones(1, 32, dtype=torch.float16),
        torch.ones(1, 32, dtype=torch.float16),
        ks,
        4,
        2,
    )
    common = dict(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seqused_k=torch.tensor([length], dtype=torch.int32),
        cmp_residual_k=torch.tensor([ratio - 1], dtype=torch.int32),
        block_table=table,
        cmp_ratio=ratio,
    )
    return args, common


@pytest.mark.parametrize("ratio", [1, 2])
def test_source_consumer_and_ordinary_use_logical_compressed_rows(modules, ratio):
    op, _ = modules
    args, common = inputs(ratio)
    ordinary = op(*args, **common)
    assert len(ordinary) == 2
    selected, _, candidates = op(*args, **common, candidate_mode=1)
    assert candidates.shape == (1, 1, 2048)
    assert set(candidates.flatten().tolist()) == set(range(8, 2056))
    assert selected.flatten().tolist() == [16447, 16446, 16445, 16444]
    result = op(*args, **common, candidate_mode=2, candidate_topk_index=candidates)
    assert torch.equal(result[0], selected)
    assert torch.equal(op(*args, **common, candidate_mode=3)[0], ordinary[0])
    restricted = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    restricted[0, 0, :4] = torch.tensor([0, 0, -1, 99999])
    result = op(*args, **common, candidate_mode=2, candidate_topk_index=restricted)
    assert result[0].flatten().tolist() == [7, 6, 5, 4]


def test_source_pins_last_visible_block_and_causal_padding(modules):
    op, _ = modules
    args, common = inputs()
    args[4].fill_(1)
    length = int(common["seqused_k"][0])
    for position in range(length - 8, length):
        args[4][common["block_table"][0, position // 64], position % 64] = 0
    _, _, blocks = op(*args, **common, candidate_mode=1)
    assert (length - 1) // 8 in blocks.flatten().tolist()
    common["seqused_k"].zero_()
    indices, _, blocks = op(*args, **common, candidate_mode=1)
    assert torch.all(indices == -1) and torch.all(blocks == -1)
    common["seqused_k"].fill_(1)
    common["cu_seqlens_q"] = torch.tensor([0, 2], dtype=torch.int32)
    q, k, w, qs, ks, topk, mode = args
    result = op(
        q.expand(2, -1, -1).contiguous(),
        k,
        w.expand(2, -1).contiguous(),
        qs.expand(2, -1).contiguous(),
        ks,
        topk,
        mode,
        **common,
        candidate_mode=3,
    )
    assert result[0][0].eq(-1).all()
    assert result[0][1, 0, 0] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(candidate_mode=0),
        dict(candidate_mode=1, candidate_topk_blocks=64),
        dict(candidate_mode=1, candidate_block_size=4),
        dict(candidate_mode=2),
    ],
)
def test_invalid_candidate_contract_rejected(modules, kwargs):
    op, _ = modules
    args, common = inputs()
    with pytest.raises(ValueError):
        op(*args, **common, **kwargs)


def test_public_route_reuses_existing_upstream_operators(modules):
    import ast
    from types import SimpleNamespace

    source = PACKAGE / "operators/indexer/quant_lightning_indexer.py"
    node = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_a3_quant_lightning_indexer"
    )
    ns = {"torch": torch, "__package__": "_qli_cpu_test.operators.indexer"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    calls = []
    indices = torch.zeros(1, 1, 4, dtype=torch.int32)
    values = torch.empty(0, dtype=torch.bfloat16)
    candidates = torch.zeros(1, 1, 2048, dtype=torch.int32)

    def make_op(name, result):
        def invoke(*args, **kwargs):
            calls.append((name, kwargs))
            return result

        return invoke

    ops = SimpleNamespace(
        quant_lightning_indexer=make_op("ordinary", (indices, values)),
        quant_lightning_indexer_candidate=make_op(
            "source", (indices, values, candidates)
        ),
        quant_sparse_lightning_indexer=make_op("consumer", (indices, values)),
    )
    args, common = inputs()
    route = ns["_a3_quant_lightning_indexer"]
    assert len(route(ops, *args, **common)) == 2
    for mode, name in ((1, "source"), (2, "consumer"), (3, "ordinary")):
        result = route(
            ops,
            *args,
            **common,
            candidate_mode=mode,
            candidate_topk_index=candidates if mode == 2 else None,
        )
        assert len(result) == 3 and calls[-1][0] == name
        assert "candidate_mode" not in calls[-1][1]
        if mode == 1:
            assert result[2] is candidates
        else:
            assert result[2].numel() == 0
    assert calls[1][1]["candidate_topk_blocks"] == 2048
    assert calls[2][1]["candidate_topk_index"] is candidates


def test_profiles_reuse_experimental_a3_ops_without_source_overlay(
    monkeypatch, tmp_path
):
    import importlib.util
    import json

    path = ROOT / "third_party/ops_transformer/build.py"
    sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
    try:
        spec = importlib.util.spec_from_file_location("_qli_build_test", path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
    finally:
        sys.path.pop(0)
    root = ROOT / "third_party/profiles/cann-9.2.0"
    main = json.loads((root / "ops_transformer.json").read_text())
    release = json.loads((root / "ops_transformer_dsv4.json").read_text())
    experiment = dict(main, **main["additional_builds"][0])
    experiment["ops"] = experiment["ops"]["arch22"]
    release["ops"] = release["ops"]["arch22"]
    for cfg in (experiment, release):
        cfg["arch_version"] = "2201"
    monkeypatch.setattr(adapter, "detect_soc_version", lambda: "ascend910_93")
    selected = adapter.configure(experiment, 1)
    assert selected["a3_indexer_bindings"] and selected["has_aicpu"]
    assert selected["commit"] == "e7da18105f08b06e01ef0fff78f439098d483746"
    assert {
        "quant_lightning_indexer_v2",
        "quant_lightning_indexer_v2_metadata",
        "quant_sparse_lightning_indexer",
    } <= set(selected["ops"])
    assert "--experimental" in selected["commands"][0]
    resolved_release = adapter.configure(release, 1)
    assert "quant_lightning_indexer_v2" not in resolved_release["ops"]
    assert not resolved_release["has_aicpu"]
    monkeypatch.setattr(adapter, "detect_soc_version", lambda: "ascend910b")
    assert adapter.configure(experiment, 1)["ops"] == ["compressor_v2"]
    assert "quant_lightning_indexer_v2" in adapter.configure(release, 1)["ops"]
    assert not list((ROOT / "third_party/ops_transformer/dsv41_indexer").rglob("*.cpp"))

    import zipfile

    wheel = tmp_path / "upstream.whl"
    prefix = "cann_ops_transformer_" + selected["vendor_name"]
    binding = """class Builder:
    def register_meta(self):
        def metadata(cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None,
                     seqused_k=None, cmp_residual_k=None):
            return torch.empty((QLI_METADATA_SIZE), dtype=torch.int32, device="npu")
        def compute(query, return_value=0):
            sparse_indices_out = torch.empty(3, dtype=torch.int32, device=query.device)
            if return_value:
                sparse_values_out = torch.empty(3, dtype=torch.bfloat16, device=query.device)
            else:
                sparse_values_out = torch.empty(0, dtype=torch.bfloat16, device=query.device)
            return (sparse_indices_out, sparse_values_out)
        return metadata, compute
"""
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            prefix + "/ops/attention/quant_lightning_indexer_v2/__init__.py", ""
        )
        archive.writestr(
            prefix
            + "/ops/attention/quant_lightning_indexer_v2/quant_lightning_indexer.py",
            binding,
        )
    package = tmp_path / "package"
    adapter.stage_python(wheel, selected, package)
    staged = (
        package
        / "_third_party"
        / selected["name"]
        / "ops/attention/quant_lightning_indexer_v2/quant_lightning_indexer.py"
    )
    ns = {"torch": torch, "QLI_METADATA_SIZE": 1024}
    exec(compile(staged.read_text(), str(staged), "exec"), ns)
    metadata, compute = ns["Builder"]().register_meta()
    query = torch.empty(1, device="meta")
    assert metadata(seqused_k=query).device.type == "meta"
    assert compute(query, 1)[1].dtype == torch.float32
    assert compute(query, 0)[1].dtype == torch.bfloat16


def test_public_wrapper_keeps_legacy_pair_and_explicit_mode_triple():
    import ast

    source = PACKAGE / "operators/indexer/quant_lightning_indexer.py"
    node = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "quant_lightning_indexer"
    )
    node.decorator_list = []
    indices = torch.tensor([[[2, 1]]], dtype=torch.int32)
    values = torch.empty(0)
    blocks = torch.tensor([[[0]]], dtype=torch.int32)
    ns = {
        "torch": torch,
        "get_impl_by_name": lambda *args: lambda *a, **kw: (indices, values, blocks),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    args, _ = inputs()
    assert len(ns["quant_lightning_indexer"](*args)) == 2
    for mode in (1, 2, 3):
        triple = ns["quant_lightning_indexer"](*args, candidate_mode=mode)
        assert len(triple) == 3 and triple[2] is blocks
