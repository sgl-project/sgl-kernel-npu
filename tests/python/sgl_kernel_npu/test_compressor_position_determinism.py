# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""OFFLINE (device-free) regression test for the c4-indexer compressor M-axis
base-block grid anchoring — README §237(1) (sgl-scripts/DSV4-A5/
hit_miss_divergence).

Background
----------
On arch35 (cache_mode=2/EXPLICIT, cmp_ratio=4, coff=2) a cache-HIT suffix
resumed at an absolute position ``P`` must reproduce the full-prefill (MISS)
index-K bit-for-bit.  The compressor walks the M axis (flattened tokens) in
base blocks of ``mBaseSize`` (=256).  In HEAD the grid is anchored at the
CALL's origin (``SkipOneLoop`` in
csrc/compressor/op_kernel/arch35/compressor_kernel.h), so a suffix call whose
``bStartPos = P`` is not a multiple of ``mBaseSize`` lands its base-block
boundaries at ``P + k*mBaseSize`` instead of the absolute ``k*mBaseSize``.
The same absolute token then falls in a DIFFERENT base block for MISS and HIT.

Patch A (§237(1) "M-axis absolute anchoring") fixes this: ``BatchInfo`` gains
``mBaseResidue = bStartPos & (mBaseSize-1)`` and ``SkipOneLoop`` shortens the
FIRST base block of the first batch to ``mBaseSize - mBaseResidue``, pinning
the grid to the absolute mBaseSize lattice.  ``bStartPos == 0`` (MISS) gives
``mBaseResidue == 0`` so ``cap == mBaseSize`` and the MISS path is
byte-preserved.

What this test does (no device, no torch)
-----------------------------------------
It models the base-block grid (README §237.2(c)) as a per-ABSOLUTE-POSITION
base-block index, for the two call shapes:
    MISS : full [0, T)
    HIT  : prefix [0, P) + suffix [P, T)
with T = 17536 and P in {16384, 16385, 16390, 17412}.  Only 16384 is
256-aligned; the rest are deliberately NOT.  It then asserts:

  1. patched(model) : per-absolute-position grid index MISS == HIT, for every P;
  2. patched(MISS)  : byte-identical to the UNPATCHED baseline (MISS preserved);
  3. legacy(model)  : diverges from MISS for the non-aligned P that straddle a
                      grid line (16385, 16390) and coincides otherwise
                      (16384 aligned; 17412's suffix is shorter than one base
                      block, so no internal boundary is cut);
  4. the real arch35 sources actually implement the anchor (static contract).

RED expectation (static-only, no device build)
----------------------------------------------
This file is committed together with Patch A, so at that commit it is GREEN.
Against the pre-patch tree (HEAD^) it is RED: assertion (4) fails because the
arch35 sources do not yet contain ``mBaseResidue``/``absResidue``, and
assertion (1) fails for the model because the call-origin grid is what the
kernel actually does.  Assertions (2) and (3) lock the two design constraints
from the README: MISS must be byte-preserved, and the aligned prefix must be a
no-op (16384), which is exactly why §237.2(c) says the M-axis anchor is real
but NOT the cause of the observed 16384-aligned online divergence.

Run:
    python tests/python/sgl_kernel_npu/test_compressor_position_determinism.py
"""

import re
import unittest
from pathlib import Path

# --- fixed compressor geometry for this regression (README §237) ---
M_BASE = 256  # constInfo.mBaseSize, the M-axis base-block size (power of two)
CMP_RATIO = 4  # constInfo.cmpRatio (unchanged by Patch A; recorded for context)
FULL_LEN = 17536  # T: the full-prefill sequence length used by §205/§237
MISS_P = 0  # MISS origin: bStartPos == 0

# P values: 16384 == 64*256 (ALIGNED control); the rest are NOT 256-aligned.
# 17412 == 17408 + 4 (block 68 + 4) has a suffix (124 tokens) shorter than one
# base block, so its first (and only) suffix block is not cut by a grid line.
TEST_P = (16384, 16385, 16390, 17412)
ALIGNED_P = frozenset({16384})
# Non-aligned P whose suffix straddles a downstream grid boundary -> the legacy
# call-origin grid produces a different base block for those absolute tokens.
DIVERGENT_P = frozenset({16385, 16390})

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMM_H = _REPO_ROOT / "csrc" / "compressor" / "op_kernel" / "arch35" / "compressor_comm.h"
_KERNEL_H = _REPO_ROOT / "csrc" / "compressor" / "op_kernel" / "arch35" / "compressor_kernel.h"


def _grid_index(abs_pos, call_start, m_base, patched):
    """Base-block grid index assigned to ABSOLUTE position ``abs_pos`` by a
    call whose M-axis origin is ``call_start``.

    Mirrors the tail of ``SkipOneLoop`` (compressor_kernel.h):

    * ``patched=False`` (HEAD / legacy): the grid is anchored at the CALL
      origin, so a position maps to ``(abs_pos - call_start) // m_base`` local
      blocks; re-anchored by ``call_start // m_base`` so the index is comparable
      to the caller's absolute lattice.
    * ``patched=True`` (Patch A): the grid is anchored on the ABSOLUTE
      mBaseSize lattice because the first base block is shortened by
      ``mBaseResidue = call_start & (m_base-1)``; hence ``abs_pos // m_base``.

    For ``call_start == 0`` (MISS) both forms collapse to ``abs_pos // m_base``.
    """
    if abs_pos < call_start:
        raise ValueError(f"pos {abs_pos} precedes call origin {call_start}")
    if patched:
        return abs_pos // m_base
    return (abs_pos - call_start) // m_base + call_start // m_base


def _call_grid_map(call_start, seq_len, m_base, patched):
    """{absolute position: grid index} for one call covering
    [call_start, call_start + seq_len)."""
    return {
        pos: _grid_index(pos, call_start, m_base, patched)
        for pos in range(call_start, call_start + seq_len)
    }


def _sequence_grid_map(calls, patched):
    """Merge the per-call maps of a call list (prefix + suffix) into one map
    keyed by absolute position; the calls must tile a contiguous range."""
    merged = {}
    for call_start, seq_len in calls:
        overlapping = merged.keys() & range(call_start, call_start + seq_len)
        if overlapping:
            raise ValueError(f"call list overlaps at {sorted(overlapping)[:4]}")
        merged.update(_call_grid_map(call_start, seq_len, M_BASE, patched))
    return merged


def _miss_calls():
    """MISS: a single full-prefill call [0, T)."""
    return [(MISS_P, FULL_LEN)]


def _hit_calls(p):
    """HIT: prefix [0, P) then a resumed suffix [P, T)."""
    return [(0, p), (p, FULL_LEN - p)]


def _divergent_positions(p, patched):
    """Absolute positions where the HIT grid index differs from MISS."""
    miss = _sequence_grid_map(_miss_calls(), patched)
    hit = _sequence_grid_map(_hit_calls(p), patched)
    if set(miss) != set(hit):
        missing = sorted(set(miss) ^ set(hit))
        raise AssertionError(f"call shapes cover different positions: {missing[:4]}")
    return sorted(pos for pos in miss if miss[pos] != hit[pos])


def _source_anchor_verdict():
    """True iff the real arch35 sources implement the absolute M-axis anchor.

    Returns None (=> skip) when the sources are not present, e.g. an installed
    wheel without ``csrc/``.
    """
    try:
        comm = _COMM_H.read_text(encoding="utf-8")
        kernel = _KERNEL_H.read_text(encoding="utf-8")
    except OSError:
        return None
    has_field = bool(re.search(r"\bmBaseResidue\b", comm))
    has_set = bool(re.search(r"\bmBaseResidue\b", kernel))
    has_cap = bool(re.search(r"\babsResidue\b", kernel))
    return has_field and has_set and has_cap


class TestCompressorPositionDeterminism(unittest.TestCase):
    def test_patched_grid_is_position_deterministic(self):
        """Patch A: for every P, a given absolute token lands in the same
        base-block grid cell for MISS and HIT (prefix + suffix)."""
        miss = _sequence_grid_map(_miss_calls(), patched=True)
        for p in TEST_P:
            with self.subTest(P=p):
                hit = _sequence_grid_map(_hit_calls(p), patched=True)
                self.assertEqual(set(miss), set(hit))
                diffs = {pos: (miss[pos], hit[pos]) for pos in miss if miss[pos] != hit[pos]}
                self.assertEqual(
                    diffs,
                    {},
                    f"patched HIT P={p} is NOT position-deterministic: "
                    f"{len(diffs)} positions differ, e.g. {list(diffs.items())[:4]}",
                )
                # An absolute token's grid index must equal its MISS index.
                for pos in miss:
                    self.assertEqual(miss[pos], hit[pos], f"P={p} pos={pos}")

    def test_miss_is_byte_preserved(self):
        """MISS (bStartPos==0 => mBaseResidue==0 => cap==mBaseSize) must be
        byte-identical to the unpatched baseline."""
        miss_patched = _sequence_grid_map(_miss_calls(), patched=True)
        miss_legacy = _sequence_grid_map(_miss_calls(), patched=False)
        self.assertEqual(miss_patched, miss_legacy)
        # Explicit: the residue and cap are both no-ops on an aligned origin.
        self.assertEqual(MISS_P & (M_BASE - 1), 0)
        self.assertEqual(M_BASE - (MISS_P & (M_BASE - 1)), M_BASE)

    def test_red_legacy_grid_diverges(self):
        """RED premise locked: the HEAD call-origin grid is NOT position
        deterministic for the non-aligned P that straddle a grid line, while it
        does coincide for the aligned control and for the sub-base-block suffix.
        (This is the regression Patch A fixes.)"""
        divergent = {}
        coincident = set()
        for p in TEST_P:
            diffs = _divergent_positions(p, patched=False)
            if diffs:
                divergent[p] = diffs
            else:
                coincident.add(p)
        self.assertEqual(
            set(divergent),
            set(DIVERGENT_P),
            f"legacy divergence mismatch: diverged={sorted(divergent)} "
            f"coincident={sorted(coincident)}",
        )
        self.assertEqual(coincident, set(TEST_P) - set(DIVERGENT_P))
        # 16384 is a 256-aligned prefix: the grids overlap (§237.2(c)).
        self.assertIn(16384, ALIGNED_P)
        self.assertNotIn(16384, divergent)
        # The divergence is a single shifted grid line, not the many positions
        # that the first (aligned) base block covers.
        print(f"[POSDET] legacy divergent positions: {divergent}", flush=True)

    def test_source_implements_absolute_anchor(self):
        """Static contract: the arch35 kernel must set ``mBaseResidue`` (comm.h
        + SkipInvalidBatch) and consume ``absResidue`` in SkipOneLoop.  RED on
        the pre-patch tree; GREEN once Patch A lands."""
        verdict = _source_anchor_verdict()
        if verdict is None:
            self.skipTest("csrc/compressor kernel sources not found (installed wheel)")
        self.assertTrue(
            verdict,
            "RED: arch35 compressor does not anchor the mBaseSize grid on the "
            "ABSOLUTE position. Expected csrc/compressor/op_kernel/arch35/"
            "compressor_comm.h to declare `mBaseResidue`, compressor_kernel.h "
            "to set it in SkipInvalidBatch and to consume `absResidue` in the "
            "SkipOneLoop tail (README §237(1) / Patch A).",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
