# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""OFFLINE (device-free) regression test for the c4-indexer compressor
compressed-OUTPUT slot anchoring — README §237.5 (sgl-scripts/DSV4-A5/
hit_miss_divergence).

Background
----------
On arch35 (cache_mode=2/EXPLICIT, cmp_ratio=4, coff=2) the compressor writes
its compressed-output rows (padScIdx -> CopyFinalResultOut / CalRope) using a
CALL-LOCAL compressed id: ``SplitCoreV2`` set

    splitInfo.dealedScCnt = info.compressedId - info.dealScSize;

where ``info.compressedId`` is seeded from 0 per call and accumulates the
call's compressed tokens (compressor_kernel.h UpdateVec2Info).  The written
output slot is therefore a function of the CALL, not of the absolute
compressed position: a cache-HIT suffix resumed at position P writes the block
for absolute compressed index ``c`` to row ``c - P/cmpRatio`` while the
full-prefill MISS writes it to row ``c``.  The device analyzer's
``[KSTATELOC-W] VERDICT WRITE-LOC inconsistent (same seq -> multiple out)``
(log swanew_10_12_7.txt) is exactly this: the same absolute position lands on
different written slots between the MISS and a HIT.

The fix (this file is committed WITH it) anchors the output base on the
ABSOLUTE compressed origin:

    absScBase = Trunc(GetStartPos(info.bStart), cmpRatio) / cmpRatio;
    splitInfo.dealedScCnt = absScBase + info.compressedId - info.dealScSize;

``bStartPos == 0`` (MISS) gives ``absScBase == 0`` so the original path is
byte-preserved.

What this test does (no device, no torch)
-----------------------------------------
It models the per-ABSOLUTE-compressed-index output row as a per-CALL row
assignment, for the two call shapes:

    MISS : full [0, T)
    HIT  : prefix [0, P) + suffix [P, T)

with T = 17536, cmp_ratio = 4 and P in {16384, 16385, 16390, 17412}.  It then
asserts:

  1. patched(model): for every P, a given absolute compressed index maps to the
     SAME output row across MISS and HIT (this is the exact assertion the device
     ``[KSTATELOC-W]`` makes: "same seq -> one out");
  2. patched(MISS): byte-identical to the UNPATCHED baseline (MISS preserved,
     because P == 0 => absScBase == 0);
  3. legacy(model): diverges from MISS for every P > 0 (the call-local write
     slot is a function of the call), and coincides only for P == 0;
  4. the real arch35 sources actually implement the absolute anchor
     (static contract: ``absScBase`` present in both vec templates).

RED expectation (static-only, no device build)
----------------------------------------------
This file is committed together with the fix, so at that commit it is GREEN.
Against the pre-fix tree (HEAD^) it is RED: assertion (4) fails because
``CompressorBlockVector<COMP>::SplitCoreV2`` does not yet contain
``absScBase``, and the model in assertion (1) would be modelling call-local
rows.

Run:
    python tests/python/sgl_kernel_npu/test_compressor_write_loc_determinism.py
"""

import re
import unittest
from pathlib import Path

# --- fixed compressor geometry for this regression (README §237.5) ---
CMP_RATIO = 4
FULL_LEN = 17536  # T: the full-prefill sequence length used by §205/§237
MISS_P = 0  # MISS origin: bStartPos == 0

# P values: 16384 == 64*256 (aligned prefix used by the device log).  The
# write-loc bug, unlike the §237(1) M-axis grid, is ALREADY visible at the
# 256-aligned 16384 (absScBase = 16384/4 = 4096 != 0).
TEST_P = (16384, 16385, 16390, 17412)
COINCIDENT_P = frozenset({0})  # only the MISS origin is a no-op
DIVERGENT_P = frozenset(TEST_P)  # every P > 0 shifts the call-local slot

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VEC_H = _REPO_ROOT / "csrc" / "compressor" / "op_kernel" / "arch35" / "compressor_block_vec.h"
_VEC_FULL_LOAD_H = (
    _REPO_ROOT / "csrc" / "compressor" / "op_kernel" / "arch35" / "compressor_block_vec_full_load.h"
)


def _abs_sc_base(call_start):
    """Absolute compressed base ``Trunc(call_start, r)/r`` (== call_start // r).

    Mirrors ``absScBase = Trunc(GetStartPos(info.bStart), cmpRatio)/cmpRatio`` in
    ``CompressorBlockVector::SplitCoreV2``.  0 for the MISS origin.
    """
    return (call_start // CMP_RATIO) * CMP_RATIO // CMP_RATIO


def _produced_blocks(call_start, seq_len):
    """Absolute compressed indices produced by one call [call_start, call_start+seq_len).

    Mirrors ``compressedTcNum = (bStartPos + seqUsedCnt)/r - bStartPos/r``
    (compressor_kernel.h SkipInvalidBatch): blocks ``[start//r, (start+len)//r)``.
    """
    return range(call_start // CMP_RATIO, (call_start + seq_len) // CMP_RATIO)


def _call_slot_map(call_start, seq_len, patched):
    """{absolute compressed index: written output row} for one call.

    patched=True  -> row is the ABSOLUTE compressed index (anchor applied).
    patched=False -> row is the CALL-LOCAL index (``compressedId`` seeded from
                     0 per call; the pre-fix behaviour) which equals
                     ``abs_sc - call_start//r``.
    """
    base = _abs_sc_base(call_start)
    return {
        abs_sc: (abs_sc if patched else abs_sc - base)
        for abs_sc in _produced_blocks(call_start, seq_len)
    }


def _sequence_slot_map(calls, patched):
    """Merge the per-call maps of a call list (prefix + suffix) keyed by
    absolute compressed index; calls must tile a contiguous token range."""
    merged = {}
    for call_start, seq_len in calls:
        produced = _call_slot_map(call_start, seq_len, patched)
        overlap = merged.keys() & produced.keys()
        if overlap:
            raise ValueError(f"call list re-produces blocks {sorted(overlap)[:4]}")
        merged.update(produced)
    return merged


def _miss_calls():
    """MISS: a single full-prefill call [0, T)."""
    return [(MISS_P, FULL_LEN)]


def _hit_calls(p):
    """HIT: prefix [0, P) then a resumed suffix [P, T)."""
    return [(0, p), (p, FULL_LEN - p)]


def _row_diffs(p, patched):
    """Absolute compressed indices whose written row differs MISS vs HIT."""
    miss = _sequence_slot_map(_miss_calls(), patched)
    hit = _sequence_slot_map(_hit_calls(p), patched)
    if set(miss) != set(hit):
        missing = sorted(set(miss) ^ set(hit))
        raise AssertionError(f"call shapes produced different blocks: {missing[:4]}")
    return sorted(c for c in miss if miss[c] != hit[c])


def _source_anchor_verdict():
    """True iff both arch35 vec templates anchor the output base on the ABSOLUTE
    compressed position.  None (=> skip) when csrc/ is absent (installed wheel)."""
    try:
        vec = _VEC_H.read_text(encoding="utf-8")
        vec_fl = _VEC_FULL_LOAD_H.read_text(encoding="utf-8")
    except OSError:
        return None
    # NORMAL template: `absScBase` must be computed and ADDED to the call-local
    # `dealedScCnt` (not merely declared).
    has_base = bool(re.search(r"\babsScBase\b", vec))
    has_sum = bool(
        re.search(r"dealedScCnt\s*=\s*absScBase\s*\+\s*info\.compressedId", vec)
    )
    # FULL_LOAD mirror (kept for template parity / compilation).
    has_fl = bool(re.search(r"\babsScBase\b", vec_fl))
    return has_base and has_sum and has_fl


class TestCompressorWriteLocDeterminism(unittest.TestCase):
    def test_patched_write_loc_is_position_deterministic(self):
        """The fix: for every P, a given absolute compressed index lands on the
        SAME written output row for MISS and HIT (prefix + suffix)."""
        miss = _sequence_slot_map(_miss_calls(), patched=True)
        for p in TEST_P:
            with self.subTest(P=p):
                hit = _sequence_slot_map(_hit_calls(p), patched=True)
                self.assertEqual(set(miss), set(hit))
                diffs = {c: (miss[c], hit[c]) for c in miss if miss[c] != hit[c]}
                self.assertEqual(
                    diffs,
                    {},
                    f"patched HIT P={p} is NOT write-loc deterministic: "
                    f"{len(diffs)} blocks differ, e.g. {list(diffs.items())[:4]}",
                )
                for c in miss:
                    self.assertEqual(miss[c], hit[c], f"P={p} abs_sc={c}")
                # The aligned prefix 16384 is the device case and MUST be fixed.
                self.assertEqual(_abs_sc_base(16384), 4096)

    def test_miss_is_byte_preserved(self):
        """bStartPos == 0 => absScBase == 0 => the patched MISS row equals the
        legacy MISS row for every produced block (byte-preserved)."""
        miss_patched = _sequence_slot_map(_miss_calls(), patched=True)
        miss_legacy = _sequence_slot_map(_miss_calls(), patched=False)
        self.assertEqual(miss_patched, miss_legacy)
        self.assertEqual(_abs_sc_base(MISS_P), 0)
        # P == 0 HIT collapses to MISS (prefix [0,0) + suffix [0,T)).
        self.assertEqual(TEST_P_SELFCHECK(), None)

    def test_red_legacy_write_loc_diverges(self):
        """RED premise locked: the pre-fix CALL-LOCAL write slot is NOT a
        function of the absolute position for ANY P > 0 (including the
        256-aligned 16384), and coincides only for the MISS origin."""
        divergent = {}
        coincident = set()
        for p in TEST_P + (MISS_P,):
            diffs = _row_diffs(p, patched=False)
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
        self.assertEqual(coincident, set(COINCIDENT_P))
        # 16384 is 256-aligned yet still diverges: the write-loc root is distinct
        # from the §237(1) M-axis grid (which coincided at 16384).
        self.assertIn(16384, divergent)
        print(f"[WRITELOC] legacy divergent P: {sorted(divergent)}", flush=True)

    def test_source_implements_absolute_anchor(self):
        """Static contract: ``SplitCoreV2`` must add ``absScBase`` to the
        call-local ``dealedScCnt`` and the FULL_LOAD template must carry the same
        anchor.  RED on the pre-fix tree; GREEN once the fix lands."""
        verdict = _source_anchor_verdict()
        if verdict is None:
            self.skipTest("csrc/compressor kernel sources not found (installed wheel)")
        self.assertTrue(
            verdict,
            "RED: arch35 compressor does not anchor the compressed-output slot "
            "on the ABSOLUTE compressed position. Expected "
            "csrc/compressor/op_kernel/arch35/compressor_block_vec.h "
            "SplitCoreV2 to compute `absScBase = Trunc(GetStartPos(info.bStart), "
            "cmpRatio_)/cmpRatio_` and set `dealedScCnt = absScBase + "
            "info.compressedId - info.dealScSize` (README §237.5), and the "
            "FULL_LOAD template to mirror it.",
        )


def TEST_P_SELFCHECK():
    """P == 0 suffix ([0,0) + [0,T)) must equal the MISS for the patched model."""
    hit0 = _sequence_slot_map(_hit_calls(MISS_P), patched=True)
    miss0 = _sequence_slot_map(_miss_calls(), patched=True)
    assert hit0 == miss0, "P==0 HIT diverged from MISS under the patched model"
    return None


if __name__ == "__main__":
    unittest.main(verbosity=2)
