"""CPU-only payload and physical-page oracles for the A5 C8 CopySFA port.

These tests do not execute an NPU kernel. They validate the independent fixture
used by the A5 numerical tests, including the exact 656-byte row payload and
the 512/513 fused-to-union-copy boundary cases.
"""

import hashlib
import json
import re
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
    operator = (
        root / "csrc" / "attention" / "fused_quant_scatter_copy_sparse_flash_attention"
    )
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
    operator = (
        root / "csrc" / "attention" / "fused_quant_scatter_copy_sparse_flash_attention"
    )
    copy_sfa_build = (operator / "CMakeLists.txt").read_text()
    registration = (operator / "legacy_registration.cpp").read_text()

    assert "fused_quant_scatter_copy_sparse_flash_attention" in copy_sfa_build
    assert "fused_quant_lightning_indexer_manage" not in copy_sfa_build
    assert "fused_quant_lightning_indexer_manage" not in registration
    assert "vllm" not in registration.lower()


def test_v2_is_additive_and_keeps_the_legacy_schema() -> None:
    root = Path(__file__).parents[1]
    operator = (
        root / "csrc" / "attention" / "fused_quant_scatter_copy_sparse_flash_attention"
    )
    registration = (operator / "legacy_registration.cpp").read_text()
    modern_adapter = (operator / "torch_adapter.cpp").read_text()
    host = (operator / "op_host" / "fused_quant_copy_sfa.cpp").read_text()
    kernel = (
        operator / "op_kernel" / "fused_quant_scatter_copy_sparse_flash_attention.cpp"
    ).read_text()
    copy_kernel = (
        operator
        / "op_kernel"
        / "fused_quant_scatter_copy_sparse_flash_attention_copy.h"
    ).read_text()

    assert registration.count('"fused_quant_scatter_copy_sparse_flash_attention("') == 1
    assert "fused_quant_scatter_copy_sparse_flash_attention_v2(" in registration
    optional_schemas = {
        "current": "Tensor? current=None",
        "positions": "Tensor? positions=None",
        "current_valid": "Tensor? current_valid=None",
        "request_slots": "Tensor? request_slots=None",
        "starts": "Tensor(d!)? starts=None",
        "ends": "Tensor(e!)? ends=None",
    }
    for schema in optional_schemas.values():
        assert schema in registration
    assert 'capabilities.insert("split_query", 1)' in registration
    assert 'capabilities.insert("ring_tail", 1)' in registration
    assert 'capabilities.insert("schema_version", 1)' in registration
    assert 'capabilities.insert("split_query", 0)' in modern_adapter
    assert 'capabilities.insert("ring_tail", 0)' in modern_adapter
    assert "query_nope must be [T, H, 512]" in host
    assert "positions must be int64[T]" in host
    assert "current_valid must be bool[T]" in host
    assert "fused_quant_copy_sfa_bf16(" in kernel
    assert "DEFINE_COPY_SFA_V2_KERNEL(fused_quant_copy_sfa_v2_bf16," in kernel
    # Derived C + tail lengths are internal scratch. The caller's
    # num_cache_tokens tensor stays read-only and must not advertise mutation.
    assert "Tensor(c!) num_cache_tokens" not in registration
    assert "effectiveCacheTokens = userWorkspace" in kernel
    assert "PrepareRingTail(pipe" in kernel
    assert "GM_ADDR actualSeqLengthsQuery, GM_ADDR derivedCacheTokens" in copy_kernel
    assert (
        "cacheGm.SetGlobalBuffer((__gm__ int32_t *)derivedCacheTokens)" in copy_kernel
    )
    assert "if (current == nullptr || positions == nullptr" in copy_kernel


def test_legacy_a5_stub_preserves_batch_scheduling():
    """Legacy codegen must not silently drop CopySFA's cross-core launch policy."""
    root = Path(__file__).parents[1]
    cmake = (root / "cmake/legacy_kernels.cmake").read_text()
    target = "fused_quant_scatter_copy_sparse_flash_attention_a5_kernel"
    creation = cmake.index(f"ascendc_library({target} STATIC")
    include = cmake.index(f"ascendc_include_directories({target}", creation)
    wiring = cmake[creation:include]
    assert f"set(_a5_copy_sfa_stub {target}_host_stub_obj)" in wiring
    assert "if(NOT TARGET ${_a5_copy_sfa_stub})" in wiring
    assert "LaunchAscendKernel=LaunchCopySfaBatchKernel" in wiring
    assert "target_compile_definitions(${_a5_copy_sfa_stub} PRIVATE" in wiring
    helper = (root / "csrc/utils/batch_kernel_launch.h").read_text()
    assert "attribute.value.schemMode = 1" in helper
    assert "aclrtLaunchKernelWithHostArgs" in helper


def _evaluate_cmake_condition(condition: str, variables: dict[str, object]) -> bool:
    condition = condition.strip()
    if " OR " in condition:
        return any(
            _evaluate_cmake_condition(part, variables)
            for part in condition.split(" OR ")
        )
    if " AND " in condition:
        return all(
            _evaluate_cmake_condition(part, variables)
            for part in condition.split(" AND ")
        )
    if condition.startswith("NOT DEFINED "):
        return condition.removeprefix("NOT DEFINED ") not in variables
    if condition.startswith("NOT EXISTS "):
        return False
    if condition.startswith("NOT "):
        return not _evaluate_cmake_condition(condition.removeprefix("NOT "), variables)
    match = re.fullmatch(r'(\w+) MATCHES "([^"]+)"', condition)
    if match:
        return re.search(match.group(2), str(variables[match.group(1)])) is not None
    match = re.fullmatch(r'(\w+) STREQUAL "([^"]+)"', condition)
    if match:
        return str(variables[match.group(1)]) == match.group(2)
    return bool(variables.get(condition, False))


def _preprocess_cmake(source: str, variables: dict[str, object]) -> str:
    active = True
    stack: list[tuple[bool, bool]] = []
    selected: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"if\((.+)\)", stripped)
        if match:
            condition = _evaluate_cmake_condition(match.group(1), variables)
            stack.append((active, condition))
            active = active and condition
            continue
        if stripped == "else()":
            parent_active, condition = stack[-1]
            active = parent_active and not condition
            continue
        if stripped == "endif()":
            active, _ = stack.pop()
            continue
        if active:
            selected.append(line)
    assert not stack
    return "\n".join(selected)


def _preprocess_cpp(source: str, defined: set[str]) -> str:
    active = True
    stack: list[tuple[bool, bool]] = []
    selected: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"#ifdef (\w+)", stripped)
        if match:
            condition = match.group(1) in defined
            stack.append((active, condition))
            active = active and condition
            continue
        match = re.fullmatch(r"#if defined\((\w+)\) \|\| defined\((\w+)\)", stripped)
        if match:
            condition = any(name in defined for name in match.groups())
            stack.append((active, condition))
            active = active and condition
            continue
        if stripped == "#else":
            parent_active, condition = stack[-1]
            active = parent_active and not condition
            continue
        if stripped == "#endif":
            active, _ = stack.pop()
            continue
        if active:
            selected.append(line)
    assert not stack
    return "\n".join(selected)


def test_a5_legacy_build_selects_required_storage_operators() -> None:
    root = Path(__file__).parents[1]
    cmake = (root / "cmake" / "legacy_kernels.cmake").read_text()
    extensions = (root / "csrc" / "pytorch_extensions.cpp").read_text()
    a5_variables = {
        "_sgl_soc_lower": "ascend950",
        "SGL_KERNEL_ARCH": "arch35",
        "SGL_KERNEL_ENABLE_A3_ONLY_OPS": False,
        "SGL_KERNEL_ENABLE_A5_ONLY_OPS": True,
        "SGL_KERNEL_A5_C8_ORIGINAL_SOURCE": "/prepared/original",
        "ASCEND_DRIVER_PATH": "/driver",
        "BUILD_CATLASS_MODULE": False,
    }
    selected_cmake = _preprocess_cmake(cmake, a5_variables)
    for source in (
        "sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp",
        "sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp",
        "fused_scatter_copy_sparse_flash_attention/op_host/fused_shm_allocator.cpp",
        "fused_scatter_copy_sparse_flash_attention/op_host/fused_shm_allocator_op.cpp",
    ):
        assert source in selected_cmake
    assert "slot_map_lookup/op_kernel/slot_map_lookup_kernel.cpp" not in selected_cmake

    selected_extensions = _preprocess_cpp(extensions, {"SGL_KERNEL_ENABLE_A5_ONLY_OPS"})
    for registration in (
        '"unidex_copy(Tensor src',
        '"fused_shm_allocator_create_and_register(int size',
        '"fused_shm_allocator_free_all(int device_id)',
        'm.impl("unidex_copy"',
        'm.impl("fused_shm_allocator_create_and_register"',
        'm.impl("fused_shm_allocator_free_all"',
    ):
        assert registration in selected_extensions
    assert 'm.impl("slot_map_lookup"' not in selected_extensions
    assert 'm.impl("shm_allocator_create_and_register"' not in selected_extensions

    a3_variables = {
        **a5_variables,
        "_sgl_soc_lower": "ascend910_9382",
        "SGL_KERNEL_ARCH": "arch22",
        "SGL_KERNEL_ENABLE_A3_ONLY_OPS": True,
        "SGL_KERNEL_ENABLE_A5_ONLY_OPS": False,
    }
    selected_a3_cmake = _preprocess_cmake(cmake, a3_variables)
    assert "unidex_copy/op_kernel/unidex_copy_kernel.cpp" in selected_a3_cmake
    assert "slot_map_lookup/op_kernel/slot_map_lookup_kernel.cpp" in selected_a3_cmake
    selected_a3_extensions = _preprocess_cpp(
        extensions, {"SGL_KERNEL_ENABLE_A3_ONLY_OPS"}
    )
    assert 'm.impl("unidex_copy"' in selected_a3_extensions
    assert 'm.impl("slot_map_lookup"' in selected_a3_extensions
    assert 'm.impl("shm_allocator_create_and_register"' in selected_a3_extensions
