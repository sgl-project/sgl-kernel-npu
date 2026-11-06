"""Provenance checks for the seven vendored DeepSeek V4.1 DSL sources."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMPL = (
    ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/cannbot-dsl/attention"
)
SOURCE = IMPL.parent / "SOURCE.md"
UPSTREAM = json.loads((IMPL.parent / "UPSTREAM.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", UPSTREAM["files"], ids=lambda entry: entry["file"])
def test_vendored_source_matches_pinned_upstream(entry):
    source = IMPL / entry["file"]
    assert source.is_file(), source
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert data.startswith(b"# Copyright (c) 2026 Huawei Technologies")


def test_source_provenance_and_license_are_packaged_with_implementation():
    provenance = SOURCE.read_text(encoding="utf-8")
    assert "UPSTREAM.json" in provenance
    assert {entry["file"] for entry in UPSTREAM["files"]} == {
        p.name for p in IMPL.glob("*.py") if p.name != "__init__.py"
    }
    license_text = (IMPL.parent / "LICENSE.cannbot-dsl").read_text(encoding="utf-8")
    assert "CANN Open Software License Agreement Version 2.0" in license_text
