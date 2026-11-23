"""Check that bilingual public API signatures match the Python wrappers."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/operators/moe/zerc_moe.py"


def _signature(function):
    args = function.args
    return (
        tuple(arg.arg for arg in args.posonlyargs),
        tuple(arg.arg for arg in args.args),
        tuple(arg.arg for arg in args.kwonlyargs),
        tuple(ast.dump(default) for default in args.defaults),
        tuple(ast.dump(default) if default else None for default in args.kw_defaults),
    )


@pytest.mark.parametrize("language", ["en", "zh"])
def test_documented_signatures_match_source(language):
    document = (ROOT / f"docs/api/{language}/operators/moe/zerc_moe.md").read_text(
        encoding="utf-8"
    )
    blocks = re.findall(r"```python\n(.*?)```", document, flags=re.DOTALL)
    assert blocks, "Documentation must include the source signatures"
    documented = ast.parse(blocks[0])
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"create_zerc_moe_buffer", "zerc_moe"}
    expected = {
        node.name: _signature(node)
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    actual = {
        node.name: _signature(node)
        for node in documented.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert set(expected) == names
    assert actual == expected
