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
            yield reference.quant_lightning_indexer, validator.validate_quant_indexer_candidates
        finally:
            for name in set(sys.modules) - before:
                if name.startswith(namespace):
                    sys.modules.pop(name, None)


def inputs(ratio=1):
    # 65 blocks, with strictly increasing scores. Reverse physical pages so
    # candidate IDs and final indices cannot accidentally be physical slots.
    length = 520
    table = torch.arange(8, -1, -1, dtype=torch.int32).view(1, -1)
    q = torch.ones(1, 32, 128, dtype=torch.int8)
    k = torch.ones(9, 64, 1, 128, dtype=torch.int8)
    ks = torch.zeros(9, 64, 1, dtype=torch.float16)
    for position in range(length):
        ks[table[0, position // 64], position % 64, 0] = (position + 1) / 1024
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
    assert len(ordinary) == 2  # Existing callers keep the original contract.
    selected, _, candidates = op(
        *args, **common, candidate_mode=1, candidate_topk_blocks=64
    )
    assert candidates.shape == (1, 1, 64)
    assert set(candidates.flatten().tolist()) == set(range(1, 65))
    assert selected.flatten().tolist() == [519, 518, 517, 516]
    result = op(
        *args,
        **common,
        candidate_mode=2,
        candidate_topk_blocks=64,
        candidate_topk_index=candidates
    )
    assert torch.equal(result[0], selected)
    assert torch.equal(op(*args, **common, candidate_mode=3)[0], ordinary[0])
    # A consumer is a strict mask: negative/duplicate/out-of-range blocks
    # cannot contribute extra positions, even if those positions score higher.
    restricted = torch.full((1, 1, 64), -1, dtype=torch.int32)
    restricted[0, 0, :4] = torch.tensor([0, 0, -1, 999])
    result = op(
        *args,
        **common,
        candidate_mode=2,
        candidate_topk_blocks=64,
        candidate_topk_index=restricted
    )
    assert result[0].flatten().tolist() == [7, 6, 5, 4]


def test_source_pins_last_visible_block_and_causal_padding(modules):
    op, _ = modules
    args, common = inputs()
    args[4].fill_(1)
    args[4][common["block_table"][0, 8], 0:8] = 0
    _, _, blocks = op(*args, **common, candidate_mode=1, candidate_topk_blocks=64)
    assert 64 in blocks.flatten().tolist()
    common["seqused_k"].zero_()
    indices, _, blocks = op(*args, **common, candidate_mode=1, candidate_topk_blocks=64)
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
        candidate_mode=3
    )
    assert result[0][0].eq(-1).all()  # First query cannot see the sole future K.
    assert result[0][1, 0, 0] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(candidate_mode=0),
        dict(candidate_mode=1, candidate_topk_blocks=63),
        dict(candidate_mode=1, candidate_block_size=4),
        dict(candidate_mode=2),
    ],
)
def test_invalid_candidate_contract_rejected(modules, kwargs):
    op, _ = modules
    args, common = inputs()
    with pytest.raises(ValueError):
        op(*args, **common, **kwargs)


def test_existing_compute_name_has_three_output_meta_contract():
    source = (
        ROOT
        / "third_party/ops_transformer/dsv41_indexer/quant_lightning_indexer_v2/torch_extension/quant_lightning_indexer.py"
    )
    namespace = "qli_schema_cpu_test"
    library = torch.library.Library(namespace, "DEF")

    class Builder:
        def __init__(self, *args, **kwargs):
            pass

        def _ensure_initialized(self):
            for schema in self.schema():
                library.define(schema)
            self.register_meta()

    stub = ModuleType("cann_ops_transformer.op_builder")
    stub.OpBuilder = Builder
    stub.get_as_library = lambda: library
    code = source.read_text(encoding="utf-8").replace(
        "cann_ops_transformer::", namespace + "::"
    )
    with patch.dict(sys.modules, {stub.__name__: stub}):
        bindings = {}
        exec(compile(code, str(source), "exec"), bindings)
    q = torch.empty(3, 32, 128, dtype=torch.int8, device="meta")
    k = torch.empty(4, 64, 1, 128, dtype=torch.int8, device="meta")
    w = torch.empty(3, 32, dtype=torch.float16, device="meta")
    ks = torch.empty(4, 64, 1, dtype=torch.float16, device="meta")
    op = getattr(torch.ops, namespace).quant_lightning_indexer
    for mode in (1, 2, 3):
        outputs = op(
            q, k, w, w, ks, 512, 2, candidate_mode=mode, candidate_topk_blocks=64
        )
        assert outputs[0].shape == (3, 1, 512)
        assert outputs[2].shape == ((3, 1, 64) if mode == 1 else (0,))
        assert outputs[2].dtype == torch.int32
        assert outputs[1].dtype == torch.bfloat16
    scores = op(q, k, w, w, ks, 512, 2, return_value=1)
    assert scores[1].shape == (3, 1, 512) and scores[1].dtype == torch.float32
    schema = str(op.default._schema)
    assert 'layout_q="BSND"' in schema and 'layout_k="BSND"' in schema
    assert "mask_mode=0" in schema
    # The exported metadata function must enter the dispatcher for fake/compile.
    metadata = bindings["quant_lightning_indexer_metadata"](
        32, 1, 128, 512, 2, seqused_k=torch.empty(1, device="meta", dtype=torch.int32)
    )
    assert metadata.shape == (1024,) and metadata.device.type == "meta"


def test_overlay_is_a3_only_hashed_and_preserves_pinned_build_rules(
    tmp_path, monkeypatch
):
    import importlib.util
    import json

    path = ROOT / "third_party/ops_transformer/build.py"
    # Load only the build adapter; no NPU runtime or native extension import.
    with patch.dict(sys.modules):
        sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
        try:
            spec = importlib.util.spec_from_file_location("_qli_build_test", path)
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
        finally:
            sys.path.pop(0)
    config = json.loads(
        (ROOT / "third_party/profiles/cann-9.2.0/ops_transformer_dsv4.json").read_text()
    )
    config["ops"] = config["ops"]["arch22"]
    config["arch_version"] = "2201"
    monkeypatch.setattr(adapter, "detect_soc_version", lambda: "ascend910_93")
    item = adapter.configure(config, 1)
    assert item["a3_indexer_overlay"]
    assert all("__pycache__" not in name for name in item["a3_indexer_overlay"])
    source = tmp_path / "source"
    for op in ("quant_lightning_indexer_v2", "quant_lightning_indexer_v2_metadata"):
        cmake = source / "attention" / op / "op_host/CMakeLists.txt"
        cmake.parent.mkdir(parents=True)
        cmake.write_text("# original pinned build rules\n# checkers/qliv2_checker.cpp")

    class StopBeforeCompiler(Exception):
        pass

    monkeypatch.setattr(
        adapter, "run", lambda *args, **kw: (_ for _ in ()).throw(StopBeforeCompiler())
    )
    with pytest.raises(StopBeforeCompiler):
        adapter.build(item, source, tmp_path)
    for op in ("quant_lightning_indexer_v2", "quant_lightning_indexer_v2_metadata"):
        contents = (source / "attention" / op / "op_host/CMakeLists.txt").read_text()
        if op == "quant_lightning_indexer_v2":
            assert "checkers/" not in contents
            assert "OP_NAME QuantLightningIndexerV2" in contents
            assert "quant_lightning_indexer_v2_def.cpp" in contents
        else:
            assert contents == "# original pinned build rules\n# checkers/qliv2_checker.cpp"
    assert (
        source
        / "attention/quant_lightning_indexer_v2_metadata/op_kernel_aicpu/quant_lightning_indexer_v2_metadata_aicpu.cpp"
    ).exists()
    # A2/A5 continue building the untouched release bindings.
    monkeypatch.setattr(adapter, "detect_soc_version", lambda: "ascend910b")
    assert "a3_indexer_overlay" not in adapter.configure(config, 1)
    config["arch_version"] = "3510"
    monkeypatch.setattr(adapter, "detect_soc_version", lambda: "ascend950")
    assert "a3_indexer_overlay" not in adapter.configure(config, 1)


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
    pair = ns["quant_lightning_indexer"](*args)
    assert len(pair) == 2 and torch.equal(pair[0], indices)
    for mode in (1, 2, 3):
        triple = ns["quant_lightning_indexer"](*args, candidate_mode=mode)
        assert len(triple) == 3 and triple[2] is blocks


def test_packaged_entrypoint_imports_only_existing_bindings():
    import ast

    extension = (
        ROOT / "third_party/ops_transformer/dsv41_indexer/quant_lightning_indexer_v2/torch_extension"
    )
    definitions = {
        node.name
        for node in ast.parse((extension / "quant_lightning_indexer.py").read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    entrypoint = ast.parse((extension / "__init__.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in entrypoint.body
        if isinstance(node, ast.ImportFrom) and node.module == "quant_lightning_indexer"
        for alias in node.names
    }
    assert imports == {"quant_lightning_indexer", "quant_lightning_indexer_metadata"}
    assert imports <= definitions
