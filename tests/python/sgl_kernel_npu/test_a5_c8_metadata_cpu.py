"""CPU interpreter arithmetic for A5 ring metadata and unchanged A3 policy.

Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0 TRITON_INTERPRET=1. These tests do
not exercise the Ascend launcher ABI, numerical attention or graph capture.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sgl_kernel_npu/sgl_kernel_npu/sparsity_driven_kv_offload"
)


@pytest.fixture(autouse=True)
def preserve_scalar_mask_semantics(monkeypatch):
    """Correct only the Ascend3.2 interpreter's false-scalar store branch.

    Its semantic._store_legacy uses `if not mask` instead of `mask is None`.
    A runtime scalar false predicate then becomes an unconditional store.
    Keep the production kernels unchanged and forward this branch to the
    builder's actual masked store, with the same pointer/value conversions.
    """
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU metadata arithmetic requires TRITON_INTERPRET=1.")
    import triton.language as tl
    from triton.language import semantic

    original = semantic._store_legacy

    def store(ptr, val, mask, boundary_check, cache, eviction, builder):
        if mask is not None and not ptr.type.is_block() and not bool(mask):
            ptr_ty = ptr.type.scalar
            element = ptr_ty.element_ty
            if element == tl.int1:
                element = tl.int8
                ptr_ty = tl.pointer_type(element, ptr_ty.address_space)
                ptr = semantic.cast(ptr, ptr_ty, builder)
            val = semantic.cast(val, element, builder)
            return tl.tensor(
                builder.create_masked_store(
                    ptr.handle, val.handle, mask.handle, cache, eviction
                ),
                tl.void,
            )
        return original(ptr, val, mask, boundary_check, cache, eviction, builder)

    monkeypatch.setattr(semantic, "_store_legacy", store)


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"c8_cpu_{name}", SOURCE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("width", [1, 4])
def test_hit_counters_exclude_tail_padding_and_track_batch_fallback(width):
    """Prefix hits exclude causal tail rows; one request can switch the whole call."""
    kernels = load("c8_metadata")
    batch, k = 4, 8
    sources = torch.tensor([10, 11, 12, 13, 100, 101, 102, -1], dtype=torch.int32)
    sources = sources.repeat(batch * width, 1)
    slots = torch.arange(k, dtype=torch.int32).repeat(batch * width, 1)
    route_misses = torch.full((batch * width,), 2, dtype=torch.int32)
    unions = torch.tensor([512, 513, 9999, 9999], dtype=torch.int32)
    prefixes = torch.full((batch,), 100, dtype=torch.int32)
    lengths = torch.full((batch,), 100 + width, dtype=torch.int32)
    valid = torch.tensor([True, True, False, True])
    sparse = torch.tensor([True, True, True, False])
    counters = torch.zeros((2, len(kernels.C8_HIT_STAT_FIELDS)), dtype=torch.int32)
    arguments = dict(
        source_ids=sources,
        device_slots=slots,
        topk_misses=route_misses,
        miss_counts=unions,
        prefix_lens=prefixes,
        seq_lens=lengths,
        valid_rows=valid,
        sparse_rows=sparse,
        counters=counters,
        layer_index=1,
    )
    kernels.record_c8_hit_statistics(**arguments)
    result = dict(zip(kernels.C8_HIT_STAT_FIELDS, counters[1].tolist()))
    assert result["calls"] == 1 and result["requests"] == 2
    assert result["queries"] == 2 * width
    assert result["prefix_selected"] == 8 * width
    assert result["prefix_misses"] == 4 * width
    assert result["tail_selected"] == (2 if width == 1 else 18)
    assert result["selected"] == result["prefix_selected"] + result["tail_selected"]
    assert result["union_misses"] == 1025 and result["max_union_misses"] == 513
    assert result["requests_gt512"] == result["fallback_calls"] == 1
    assert result["union_257_512"] == result["union_513_1024"] == 1
    assert not counters[0].count_nonzero()
    unions[1] = 512
    kernels.record_c8_hit_statistics(**arguments)
    result = dict(zip(kernels.C8_HIT_STAT_FIELDS, counters[1].tolist()))
    assert result["calls"] == 2 and result["fallback_calls"] == 1
    assert result["max_union_misses"] == 513
    assert result["union_misses"] == 2049


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("full_visible", [False, True])
@pytest.mark.parametrize("first", [8191, 8192, 8319, 8320])
def test_pre_metadata_page_crossing(width, full_visible, first):
    kernel = load("c8_metadata")._prepare_kernel
    rows, batch, capacity, cols = 4, 3, 8192, 80
    requests = torch.tensor([1, 2, 0], dtype=torch.int32)
    seq = torch.tensor([first, first + 128, 0], dtype=torch.int32)
    positions = (
        torch.stack([torch.arange(width) + n for n in (first, first + 128, 0)])
        .long()
        .flatten()
    )
    last = torch.tensor([-1, 8192, -1, -1, -1], dtype=torch.int32)
    table = torch.arange(batch * cols, dtype=torch.int32).reshape(batch, cols)
    output = torch.full((batch, cols + 16), -999, dtype=torch.int32)
    specs = [
        torch.int32,
        torch.int32,
        torch.int32,
        torch.bool,
        torch.bool,
        torch.int32,
        torch.int32,
        torch.int32,
    ]
    out = [torch.empty(batch, dtype=dtype) for dtype in specs]
    kernel[(1, batch)](
        seq,
        positions,
        seq,
        requests,
        last,
        table,
        output,
        *out,
        rows,
        capacity,
        width,
        True,
        full_visible,
        False,
        262144,
        262144,
        cols,
        cols + 16,
        cols,
        1,
        cols + 16,
        1,
        128,
    )
    safe, lengths, prefixes, sparse, valid, state, next_prefix, cache = out
    expected_lengths = torch.tensor(
        [first + width, first + 128 + width, 0], dtype=torch.int32
    )
    expected_prefix = (
        expected_lengths
        if full_visible
        else (expected_lengths - width).clamp_min(0) // 128 * 128
    )
    assert lengths.tolist() == expected_lengths.tolist()
    assert prefixes.tolist() == expected_prefix.tolist()
    assert safe.tolist() == [1, 2, rows]
    assert valid.tolist() == [True, True, False]
    for b in range(batch):
        live = b < 2 and expected_prefix[b] >= capacity
        assert bool(sparse[b]) == live
        expected_state = (
            (-1 if b == 0 and expected_prefix[b] >= 8192 else -2) if live else -3
        )
        assert state[b] == expected_state
        assert next_prefix[b] == (expected_prefix[b] if live else -1)
        assert cache[b] == (capacity if live else 0)
    torch.testing.assert_close(output[:, :cols], table, rtol=0, atol=0)
    assert not output[:, cols:].count_nonzero()


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("first", [8192, 8319, 8320])
def test_post_metadata_rotates_physical_ring_only(width, packed, first):
    kernel = load("c8_metadata")._lim_copy_sfa_metadata_kernel
    batch, capacity, pages = 3, 8192, 80
    hp = capacity // 128 + (2 if packed else 0)
    source = torch.full((batch * width, 2048), 7, dtype=torch.int32)
    device_slots = torch.full_like(source, 77)
    misses = torch.full((batch * width,), 17, dtype=torch.int32)
    counts = torch.full((batch,), 19, dtype=torch.int32)
    valid = torch.tensor([True, True, False])
    sparse = torch.tensor([True, False, False])
    slots = torch.tensor([1, 2, 4], dtype=torch.int32)
    final = torch.tensor([first + width, 2000, 0], dtype=torch.int32)
    prefix = torch.tensor([first // 128 * 128, 1920, 0], dtype=torch.int32)
    kv, cache = torch.empty_like(final), torch.empty_like(final)
    hbm = torch.full((batch, hp), -1, dtype=torch.int32)
    dram = torch.full((batch, pages), -1, dtype=torch.int32)
    host = (
        torch.arange(batch * pages, dtype=torch.int32)
        .reshape(batch, pages)
        .flip(1)
        .contiguous()
    )
    kernel[(1, batch)](
        source,
        device_slots,
        misses,
        counts,
        valid,
        sparse,
        slots,
        kv,
        cache,
        hbm,
        dram,
        host,
        final,
        prefix,
        2048,
        width,
        capacity,
        hp,
        pages,
        2048,
        True,
        pages,
        pages,
        1,
        packed,
        True,
    )
    assert kv.tolist() == [int(final[0]) if packed else capacity + width - 1, 2000, 0]
    assert cache.tolist() == [
        capacity + int(final[0] - prefix[0]) if packed else capacity,
        0,
        0,
    ]
    for b in range(batch):
        base = int(slots[b]) * hp if valid[b] else 0
        expected = list(range(base, base + hp))
        if packed:
            expected[-2:] = [
                base + capacity // 128 + (int(prefix[b]) // 128 + i) % 2
                for i in range(2)
            ]
        assert hbm[b].tolist() == expected
    assert counts.tolist() == [19, 0, 0]
    assert misses.tolist() == [17] * width + [0] * (2 * width)
    assert torch.all(source[2 * width :] == -1)
    assert torch.all(device_slots[: 2 * width] == 77)
    assert torch.all(device_slots[2 * width :] == -1)
    torch.testing.assert_close(dram[:2], host[:2], rtol=0, atol=0)
    assert not dram[2].count_nonzero()
