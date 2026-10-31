"""CPU-only payload and physical-page oracles for the A5 C8 CopySFA port.

These tests do not execute an NPU kernel. They validate the independent fixture
used by the A5 numerical tests, including the exact 656-byte row payload and
the 512/513 fused-to-union-copy boundary cases.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from fused_quant_scatter_copy_sparse_flash_attention_golden import (
    PACKED_DIM,
    POISON,
    TOPK,
    make_test_data,
    physical_rows,
    unpack_rows,
)


@pytest.mark.parametrize("miss_count", [0, 512, 513, TOPK])
def test_payload_oracle_uses_physical_page_tables(miss_count: int) -> None:
    data = make_test_data(miss_count, seed=20261031)
    selected_sources = data.topk_src_ids[0, 0]
    destination_slots = data.topk_dst_slots[0, 0]
    source_rows = physical_rows(data.dram_block_table, selected_sources)
    destination_rows = physical_rows(data.hbm_block_table, destination_slots)

    dram_rows = data.dram.view(-1, PACKED_DIM)
    initial_rows = data.initial_hbm.view(-1, PACKED_DIM)
    expected_rows = data.expected_hbm.view(-1, PACKED_DIM)

    # All 656 bytes, including FP8 values, BF16 RoPE and FP32 scales, move as
    # one opaque row through independently permuted source/destination tables.
    torch.testing.assert_close(
        expected_rows[destination_rows], dram_rows[source_rows], rtol=0, atol=0
    )
    torch.testing.assert_close(
        initial_rows[destination_rows[miss_count:]],
        dram_rows[source_rows[miss_count:]],
        rtol=0,
        atol=0,
    )
    if miss_count:
        assert torch.all(initial_rows[destination_rows[:miss_count]] == POISON)


def test_cpu_attention_golden_is_derived_from_packed_payload() -> None:
    data = make_test_data(513, seed=20261031)
    sources = data.topk_src_ids[0, 0]
    rows = physical_rows(data.dram_block_table, sources)
    key, value = unpack_rows(data.dram.view(-1, PACKED_DIM)[rows])
    expected = (
        torch.softmax(data.query[0].float() @ key.T * data.scale_value, dim=-1) @ value
    ).unsqueeze(0)
    torch.testing.assert_close(data.attention_golden, expected, rtol=0, atol=0)


def test_physical_tables_are_not_identity_mappings() -> None:
    data = make_test_data(512, seed=20261031)
    for table in (data.hbm_block_table, data.dram_block_table):
        identity = torch.arange(table.size(1), dtype=torch.int32).view(1, -1)
        assert not torch.equal(table, identity)


def test_imported_payload_matches_recorded_provenance() -> None:
    root = Path(__file__).parents[1]
    operator = root / "csrc" / "fused_quant_scatter_copy_sparse_flash_attention"
    provenance = json.loads((operator / "UPSTREAM_PROVENANCE.json").read_text())
    adapted = {
        entry["path"]: entry["ported_sha256"] for entry in provenance["adapted_files"]
    }
    for entry in provenance["files"]:
        expected = adapted.get(entry["path"], entry["sha256"])
        actual = hashlib.sha256(
            (operator / "op_kernel" / entry["path"]).read_bytes()
        ).hexdigest()
        assert actual == expected, entry["path"]


def test_a5_copy_sfa_registration_has_no_lim_or_vllm_dependency() -> None:
    root = Path(__file__).parents[1]
    cmake = (root / "csrc" / "CMakeLists.txt").read_text()
    start = cmake.index("# C8 CopySFA is A5-only")
    end = cmake.index("# create shared library", start)
    copy_sfa_build = cmake[start:end]
    registration = (root / "csrc" / "a5_copy_sfa_registration.cpp").read_text()

    assert "fused_quant_scatter_copy_sparse_flash_attention" in copy_sfa_build
    assert "fused_quant_lightning_indexer_manage" not in copy_sfa_build
    assert "fused_quant_lightning_indexer_manage" not in registration
    assert "vllm" not in registration.lower()
