"""The migrated precision suite must exercise public operator entry points."""

import ast
from pathlib import Path

import pytest

SUITE = Path(__file__).parent / "dsv41_cannbot_dsl"
CASES = {
    "indexer_prologue_k": "indexer",
    "mixed_quant_sparse_flash_mla": "attention",
    "mixed_quant_sparse_flash_mla_metadata": "attention",
    "quant_lightning_indexer": "indexer",
    "quant_lightning_indexer_metadata": "indexer",
    "quant_sparse_lightning_indexer": "indexer",
    "quant_sparse_lightning_indexer_metadata": "indexer",
}


@pytest.mark.parametrize("name,category", CASES.items())
def test_precision_case_uses_public_operator(name, category):
    source = (SUITE / f"test_{name}.py").read_text(encoding="utf-8")
    imports = [
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        and node.module == f"sgl_kernel_npu.operators.{category}"
        for alias in node.names
    ]
    assert name in imports
    assert 'pytest.importorskip("torch_npu")' in source
    assert 'pytest.importorskip("cannbotdsl")' in source


def test_precision_suite_requires_ascend950():
    source = (SUITE / "conftest.py").read_text(encoding="utf-8")
    assert "Ascend950PR" in source
    assert "Ascend950DT" in source
