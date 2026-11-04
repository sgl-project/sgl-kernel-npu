"""Provenance checks for the seven vendored DeepSeek V4.1 DSL sources."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMPL = (
    ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/implementation/cannbot-dsl/attention"
)
SOURCE = IMPL.parent / "SOURCE.md"
SHA256 = {
    "indexer_prologue_k": "132b87236afd8efcea3b0263a2a68e391e100a53ff32d2eddd255b99374a38b5",
    "mixed_quant_sparse_flash_mla": "75a84000763785e5db4211204357f33a46e2b19206a7dc72fdae9c6008cab43c",
    "mixed_quant_sparse_flash_mla_metadata": "51e13bd02d35704f99298dd9fdff36ddd82c941a8106c07d8e668d29664ad40a",
    "quant_lightning_indexer": "e2dd446eabc64b051f2568aaf114eb91dc93bb807a5a60efd2b0df9a7faaecae",
    "quant_lightning_indexer_metadata": "d78f6062c2c8897161cf946706f46a7a519db19047d4c53a9e8b05b62a9b0ed0",
    "quant_sparse_lightning_indexer": "b4795059c3c016180431928e3ae0a9ae2cacbd7346145dc6fc26abc62ac13568",
    "quant_sparse_lightning_indexer_metadata": "c75be92de66a0307500bb5da12f622ad35df87f2dd9bee15a09bbc4d3a19a0a4",
}


@pytest.mark.parametrize("name,digest", SHA256.items())
def test_vendored_source_matches_pinned_upstream(name, digest):
    source = IMPL / f"{name}.py"
    assert source.is_file(), source
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    assert data.startswith(b"# Copyright (c) 2026 Huawei Technologies")


def test_source_provenance_and_license_are_packaged_with_implementation():
    provenance = SOURCE.read_text()
    assert "71ee7fdffa34c3e8a153fa55a0c970bb2d3a3231" in provenance
    for digest in SHA256.values():
        assert digest in provenance
    license_text = (IMPL.parent / "LICENSE.cannbot-dsl").read_text()
    assert "CANN Open Software License Agreement Version 2.0" in license_text
