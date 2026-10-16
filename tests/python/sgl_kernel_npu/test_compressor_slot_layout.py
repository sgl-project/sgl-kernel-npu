# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import unittest

import numpy as np
import torch
import torch_npu

from test_compressor import _is_arch35, _make_inputs, _reference_compressor

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

# Kernel-side stateLoc diagnostic (sibling task bg_fc11899d). The host reads
# SGL_DSV4_STATELOC_DUMP once at tiling time (csrc/compressor/op_host/arch35/
# compressor_tiling.cpp) and gates [KSTATELOC] printf lines in ReadState
# (op=L), ReadFromCacheState (op=R) and WriteToCacheState (op=W) of
# arch35/compressor_block_vec.h. Set at import time so it is in effect before
# the first op build in this process: one device run yields BOTH the slot-layout
# reproduction AND the kernel's actual stateLoc/tableColumn for run A vs run B
# (grep '[KSTATELOC]' from the captured log).
os.environ.setdefault("SGL_DSV4_STATELOC_DUMP", "1")

# --- fixed decode-step shape: the decode step that diverges online ---
# README sgl-scripts/DSV4-A5/hit_miss_divergence §205/§206/§213/§220/§225:
# block 136 = positions [17408, 17536); first divergent state row at position
# 17534 (ring residue 6), then 17535. The decode call that covers 17534 has
# batch=1, start_pos=17534, seqused=1, cu_seqlens=[0,1].
START_POS = 17534
CMP_START_POS = 17535  # the decode step that DOES produce cmp_kv (block-136)
SEQUSED = 1
COFF = 2          # OVERLAP
CMP_RATIO = 4
HEAD_DIM = 128    # indexer c4 (idx=1) width -- the layer that diverges (§214)
HIDDEN = 1024
RING_SIZE = 8     # c4 state block size == ring size (state_cache.shape[1])
ROW_LEN = 2 * COFF * HEAD_DIM  # 512 floats per state row

# Physical flat slots observed on device (§205/§214): block-136 state rows live
# at 1104..1111 on the miss path (SWA page 138) and at 1184..1191 on the hit
# path (SWA page 148) -- +80 = +10 SWA pages. This +80 is the ONLY variable
# this test varies; everything else (x, weights, carry content, call shape) is
# bit-identical across the two runs.
MISS_BASE = 1104
HIT_BASE = 1184
ODD_BASE = MISS_BASE + RING_SIZE  # 1112 = +8 = one SWA page, ODD page parity

SEED = 20261010
CONTENT_SCALE = 1.0  # carry/state row magnitude so a wrong-slot read is O(1)

# Real-input loader: env var pointing at a device-dumped .npz (see docstring).
ENV_INPUTS = "SGL_DSV4_SLOTTEST_INPUTS"
# Optional batch=2 cross-check sharing ONE state_cache (see docstring).
ENV_BATCH2 = "SGL_DSV4_SLOTTEST_BATCH2"

# Golden-compare tolerances (CPU reference vs NPU op). The state write-back
# matches the CPU reference to ~1e-8 (fp32, README §211); the cmp_kv path is
# looser because softmax/rms_norm accumulate bf16 error (test_compressor.py
# ring cases assert < 0.05). A wrong-slot read/write moves by O(1), so these
# bounds separate "match" from "bug" by two orders of magnitude.
STATE_TOL = 1e-2
CMP_TOL = 5e-2

# max-abs-diff threshold separating "LOGIC" (online ~2e-3 element / ~27 sum)
# from "numerical ~1e-8" (T5 K-split). Reported raw so the reader can judge.
LOGIC_MAXABS = 1e-3


def _build_slot_layout_table(base_slots, start_pos, seqused, coff, cmp_ratio, ring_size):
    """EXPLICIT state_block_table for the fixed decode call shape, one row per
    batch (base_slots and start_pos are parallel lists).

    The kernel indexes the table at column
        tableColumn = coff*cmp_ratio + pos - start_pos   (arch35
        compressor_block_vec.h ReadFromCacheState/WriteToCacheState)
    i.e. column c -> position (start_pos - history + c). Each position maps to
    its page row: slot = base_slot + (pos % ring_size), reproducing the
    production mapping state_loc = swa_page*ring_size + swa_loc % ring_size
    with swa_page = base_slot // ring_size (138 for miss, 148 for hit).

    This is the same convention the CPU reference consumes via
    _explicit_state_loc (table_column = history + seq_idx - batch_start_pos),
    so the SAME table drives both the op and _reference_compressor.
    """
    history_size = coff * cmp_ratio
    capacity = seqused
    batch = len(base_slots)
    table = torch.zeros((batch, history_size + capacity), dtype=torch.int32)
    for b in range(batch):
        for column in range(history_size + capacity):
            position = start_pos[b] - history_size + column
            table[b, column] = base_slots[b] + (position % ring_size)
    return table


def _make_absolute_slot_state(block_num, ring, row_len, gen):
    """state content as a deterministic function of the ABSOLUTE slot index:
    state[slot] = f(slot) (seeded), NOT a +80-translated copy of the same rows.

    This is the (c-i) fix for the design blind spot: with content keyed to the
    absolute slot, a FIXED-relative-address bug (read/write base+k instead of
    the table's absolute slot) no longer cancels between run A (miss, base
    1104) and run B (hit, base 1184) -- f(1104+k) != f(1184+k), so the bug
    shows up as an op-vs-reference mismatch (and bit-identical A/B/C outputs).
    """
    total = block_num * ring
    content = torch.randn(total, row_len, generator=gen).float() * CONTENT_SCALE
    return content.view(block_num, ring, row_len)


def _place_state_rows(state, base_slot, rows, ring, row_len):
    """Overwrite the `ring` rows at absolute slots base_slot..base_slot+ring-1
    with `rows` (shape [ring, row_len]). Used to inject real device-dumped
    carry rows into an otherwise synthetic f(slot) state."""
    flat = state.view(-1, row_len)
    flat[base_slot:base_slot + ring] = rows
    return state


def _load_real_inputs(path):
    """Load real tensors dumped from a device run (README §226-b) from an .npz.

    Every key is optional; a missing key falls back to the seeded synthetic
    value and the caller prints a LOUD warning. Returns a dict keyed by the
    canonical names below (only the keys present in the npz are included).

    .npz schema (the device must dump these, coordinate with [KSTATELOC] /
    SGL_DSV4_STATELOC_DUMP in compressor_block_vec.h):
        x            [1, hidden]   decode-token hidden state for position 17534
                                   (hidden=1024).
        x_cmp        [1, hidden]   decode-token hidden state for position 17535
                                   (the cmp_kv step); required to golden-check
                                   the cmp_kv step with real inputs.
        wkv          [coff*head_dim, hidden] = [256, 1024]
        wgate        [coff*head_dim, hidden] = [256, 1024]
        ape          [cmp_ratio, coff*head_dim] = [4, 256]
        norm_weight  [head_dim] = [128]
        rope_sin     [rows, 64]
        rope_cos     [rows, 64]
        state_1104   [8, 512]   full state rows for absolute slots 1104..1111
        state_1184   [8, 512]   full state rows for absolute slots 1184..1191
        state_1112   [8, 512]   (optional) absolute slots 1112..1119 (odd page)
    """
    data = np.load(path, allow_pickle=False)
    out = {}

    def _tensor(key, dtype):
        return torch.from_numpy(np.asarray(data[key])).to(dtype).clone()

    # On-device dtypes: x/weights are bf16, state/ape/norm/rope are fp32.
    # Normalizing here (parse at the boundary) makes the dump's saved dtype
    # irrelevant -- a bf16 value round-tripped through fp32 recovers exactly.
    for key in ("x", "x_cmp", "wkv", "wgate"):
        if key in data:
            out[key] = _tensor(key, torch.bfloat16)
    for key in ("ape", "norm_weight", "rope_sin", "rope_cos",
                "state_1104", "state_1184", "state_1112"):
        if key in data:
            out[key] = _tensor(key, torch.float32)
    return out


def _op_decode(p, state_cpu, table):
    """Run ONE decode step on the NPU op; returns (cmp_kv_out.cpu(),
    state_out.cpu()). Mirrors ascend_dsv4_backend.py:839-858: NOTE it does NOT
    pass state_cache_stride_dim0 (the online call omits it), so this test drops
    it too (README §226-6ii)."""
    state_npu = state_cpu.clone().npu()
    out = torch.ops.npu.compressor(
        p["x"].npu(),
        p["wkv"].npu(),
        p["wgate"].npu(),
        state_npu,
        p["ape"].npu(),
        p["norm_weight"].npu(),
        rope_sin=p["rope_sin"].npu(),
        rope_cos=p["rope_cos"].npu(),
        rope_head_dim=64,
        cmp_ratio=CMP_RATIO,
        state_block_table=table.npu(),
        cu_seqlens=p["cu_seqlens"].npu() if p["cu_seqlens"] is not None else None,
        seqused=torch.tensor(p["seqused"], dtype=torch.int32).npu(),
        start_pos=torch.tensor(p["start_pos"], dtype=torch.int32).npu(),
        coff=COFF,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=2,
    )
    torch_npu.npu.synchronize()
    return out.cpu(), state_npu.cpu()


def _ref_decode(p, state_cpu, table):
    """CPU reference (_reference_compressor) for the same decode step, over the
    same state tensor and table. Returns (ref_cmp_kv, ref_mask, ref_state). The
    reference mutates its kv/score halves in place, so it is fed clones and the
    two halves are re-joined into the [..., 2*coff*head_dim] state_cache form."""
    ww = COFF * HEAD_DIM
    kv_state = state_cpu[..., :ww].clone()
    score_state = state_cpu[..., ww:].clone()
    ref_cmp_kv, ref_mask = _reference_compressor(
        p["x"], p["wkv"], p["wgate"], kv_state, score_state,
        torch.zeros_like(kv_state, dtype=torch.bool),
        torch.zeros_like(score_state, dtype=torch.bool),
        p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
        block_table=table,
        cu_seqlens=p["cu_seqlens"].tolist() if p["cu_seqlens"] is not None else None,
        seqused=p["seqused"],
        start_pos=p["start_pos"],
        rope_head_dim=64,
        cmp_ratio=CMP_RATIO,
        coff=COFF,
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=2,
    )
    ref_state = torch.cat([kv_state, score_state], dim=-1)
    return ref_cmp_kv, ref_mask, ref_state


def _state_maxabs(a, b):
    return float((a - b).abs().max())


def _cmp_maxabs(op_out, ref_out, ref_mask):
    """max-abs-diff of the cmp_kv output over the reference's valid entries.
    Returns NaN when the step produces no cmp_kv (ref mask empty) or the op
    returned an empty/unshaped tensor."""
    mask_np = np.asarray(ref_mask)
    if not mask_np.any() or op_out.numel() == 0 or tuple(op_out.shape) != tuple(ref_out.shape):
        return float("nan")
    mask_t = torch.from_numpy(mask_np)
    return float((op_out.float() - ref_out.float()).abs()[mask_t].max())


class TestCompressorSlotLayout(unittest.TestCase):
    def test_decode_slot_layout_invariance(self):
        """Does the c4 compressor depend on the absolute state SLOT (page
        boundary) when the decode call shape is held fixed?

        This is NOT T5 (test_compressor_resume_boundary_state): T5 varies the
        CALL shape (full prefill [0,17536) vs prefix+suffix [16384,17536)) and
        only reproduces the ~1e-8 numerical (K-split) phenomenon -- the wrong
        one (§209-§213). Here the call shape is FIXED to the decode step that
        diverges online (start_pos=17534, seqused=1) and the ONLY variable is
        which physical SLOTS the carry content sits at, driven by the
        state_block_table:
          Run A (miss):  slots 1104..1111 (SWA page 138).
          Run B (hit):   slots 1184..1191 (SWA page 148, +80 = +10 pages).
          Run C (odd):   slots 1112..1119 (SWA page 139, +8 = page PARITY flip).
        x / wkv / wgate / ape / norm_weight / rope / cu_seqlens / seqused /
        start_pos are shared bit-for-bit across all runs.

        ---- (c-i) content = f(absolute slot) ----
        The state content is generated as a DETERMINISTIC function of the
        ABSOLUTE slot index (seeded), NOT as a +80-translated copy of the same
        rows. This closes the design blind spot of the prior version: with a
        translated copy, a FIXED-relative-address bug (read/write base+k
        instead of the table's absolute slot) errs identically in A and B and
        cancels to a false IDENTICAL. With f(slot), f(1104+k) != f(1184+k), so
        a relative-address bug becomes visible.

        ---- (c-iii) golden compare ----
        The A/B/C outputs are NOT compared to each other as the verdict (with
        f(slot) a CORRECT op also differs across layouts). The verdict is the
        golden compare against the existing CPU reference _reference_compressor
        (test_compressor.py), which implements the faithful position->absolute
        slot->f(slot) mapping. Only "op == reference on every layout" proves
        the op resolves absolute slots correctly.

        ---- (a) cmp_kv step ----
        A single 17534%4==2 decode completes no cmp_ratio group, so the host
        returns an empty cmp_kv (uninitialized, not meaningful). A second FIXED
        decode step start=17535, seqused=1 (NOT seqused=2, which would deviate
        from the online single-token shape) DOES produce the block-136 cmp_kv
        (global index-K block 4383, cmp_kv[-1]); its READ of carry positions
        17528..17534 is where a read-side slot bug shows, so it is compared
        too.

        VERDICT MATRIX (printed, primary = golden compare):
          * op == reference on A AND B AND C (state within 1e-2, cmp within
            5e-2)  => SLOT-INVARIANT: op resolves absolute slots faithfully;
            the hidden input (if the online ~27 divergence persists) is OUTSIDE
            the op (pool/framework/gamma).
          * op_A == reference but op_B (or op_C) != reference  => ABSOLUTE-SLOT
            sensitivity: the op mis-resolves slots at the +80 (or +8) offset --
            (alpha) read / (gamma) write confirmed.
          * op_A == op_B == op_C bit-identical yet != reference  => RELATIVE-
            ADDRESS bug: the op reads/writes a FIXED relative offset and ignores
            the absolute slot -- exactly the bug the +80 translation used to
            hide.
          * op_A == ref_A, op_B == ref_B, op_C != ref_C  => PAGE-PARITY bug:
            correct on even pages, wrong on the odd page (+8).
          * everything ~1e-8 only  => did NOT capture the hidden input (Q2 bar:
            must reach the online ~27, not ~1e-8).

        CAVEAT (synthetic inputs): a synthetic-input "all match reference" does
        NOT prove online slot-invariance -- §207->§209 showed synthetic
        'no-repro' is untrustworthy. Only REAL tensors (SGL_DSV4_SLOTTEST_INPUTS
        .npz, see _load_real_inputs) make a MATCHING verdict meaningful; a
        synthetic IDENTICAL is necessary but not sufficient evidence.

        REAL-INPUT loader: env SGL_DSV4_SLOTTEST_INPUTS=<npz> feeds real x,
        wkv/wgate, ape, norm_weight, rope_sin/cos and the full 512-element
        state rows for 1104..1111 / 1184..1191 (and 1112..1119). If unset, the
        test falls back to seeded synthetic and PRINTS A LOUD WARNING. The exact
        tensors the device must dump are documented in _load_real_inputs.

        KSTATELOC: the test sets SGL_DSV4_STATELOC_DUMP=1 at import; capture the
        run and `grep '\\[KSTATELOC\\]' <log>` to read the kernel's actual
        stateLoc/tableColumn for run A vs run B (op=L read-left, op=R read,
        op=W write).

        Op call shape + _make_inputs usage mirrored from test_compressor.py:
        the torch.ops.npu.compressor(...) argument order/attrs are copied from
        test_compressor_resume_boundary_state (test_compressor.py:1089-1110) and
        _run_case (test_compressor.py:708-728); head_dim=128/hidden=1024 mirror
        that same test (test_compressor.py:1051). Only the state cache content
        and the state_block_table are replaced with the slot-layout forms. The
        _make_inputs(...) call is kept byte-aligned to the real signature and
        MUST be confirmed at first device run (py_compile passing is NOT the
        same as the op accepting the shapes).
        """
        if not _is_arch35():
            self.skipTest("A5 (arch35) EXPLICIT state layout only")

        # --- inputs: decode 17534 + cmp_kv 17535 (same weights, different x) ---
        base = _make_inputs(
            [START_POS],
            SEQUSED,
            COFF,
            CMP_RATIO,
            HEAD_DIM,
            HIDDEN,
            2,
            "TH",
            torch.bfloat16,
            1,
            16,
            ring_size=RING_SIZE,
            total_seq=START_POS + SEQUSED,
        )
        p_cmp = _make_inputs(
            [CMP_START_POS],
            SEQUSED,
            COFF,
            CMP_RATIO,
            HEAD_DIM,
            HIDDEN,
            2,
            "TH",
            torch.bfloat16,
            1,
            16,
            ring_size=RING_SIZE,
            total_seq=CMP_START_POS + SEQUSED,
            seed=SEED,
        )
        for name in ("wkv", "wgate", "ape", "norm_weight", "rope_sin", "rope_cos"):
            p_cmp[name] = base[name]

        block_num = base["state_cache"].shape[0]
        ring = base["state_cache"].shape[1]
        self.assertEqual(ring, RING_SIZE)

        # --- real inputs (optional) ---
        real = {}
        env_path = os.environ.get(ENV_INPUTS)
        if env_path:
            real = _load_real_inputs(env_path)

        def _apply(name, default):
            return real.get(name, default)

        base["x"] = _apply("x", base["x"])
        for name in ("wkv", "wgate", "ape", "norm_weight", "rope_sin", "rope_cos"):
            base[name] = _apply(name, base[name])
        p_cmp["x"] = real.get("x_cmp", p_cmp["x"])
        for name in ("wkv", "wgate", "ape", "norm_weight", "rope_sin", "rope_cos"):
            p_cmp[name] = base[name]

        # --- state content = f(absolute slot) ---
        gen = torch.Generator().manual_seed(SEED)
        state = _make_absolute_slot_state(block_num, ring, ROW_LEN, gen)
        if "state_1104" in real:
            state = _place_state_rows(state, MISS_BASE, real["state_1104"], ring, ROW_LEN)
        if "state_1184" in real:
            state = _place_state_rows(state, HIT_BASE, real["state_1184"], ring, ROW_LEN)
        if "state_1112" in real:
            state = _place_state_rows(state, ODD_BASE, real["state_1112"], ring, ROW_LEN)

        synthetic = not env_path
        missing = [k for k in ("x", "wkv", "wgate", "ape", "norm_weight",
                               "rope_sin", "rope_cos", "state_1104", "state_1184")
                   if k not in real]
        if synthetic:
            print(
                "\n[SLOTLAYOUT-WARNING] "
                f"{ENV_INPUTS} is UNSET -> running on SEEDED SYNTHETIC inputs. "
                "A synthetic IDENTICAL (or synthetic all-match-reference) is "
                "NOT trustworthy evidence of online slot-invariance "
                "(README §207->§209): the real carry/weights may exercise a "
                "path the synthetic does not. Dump real tensors and set the "
                "env var to make a MATCHING verdict meaningful.\n",
                flush=True,
            )
        elif missing:
            print(
                f"[SLOTLAYOUT-WARNING] {ENV_INPUTS} provided but missing keys "
                f"{missing} -> those fall back to synthetic.\n",
                flush=True,
            )

        # --- tables ---
        table_A = _build_slot_layout_table(
            [MISS_BASE], [START_POS], SEQUSED, COFF, CMP_RATIO, RING_SIZE
        )
        table_B = _build_slot_layout_table(
            [HIT_BASE], [START_POS], SEQUSED, COFF, CMP_RATIO, RING_SIZE
        )
        table_C = _build_slot_layout_table(
            [ODD_BASE], [START_POS], SEQUSED, COFF, CMP_RATIO, RING_SIZE
        )

        # Sanity: tables differ only by the base offset; and (for the SYNTHETIC
        # f(slot) content only) the content at the two bases genuinely differs,
        # so a relative-address bug is visible. Real dumped carry may be
        # legitimately identical across miss/hit (README §220: carry positions
        # 17528..17533 match online), so the non-degeneracy guard is skipped
        # when real state rows were injected.
        self.assertEqual(int((table_B - table_A).unique().item()), HIT_BASE - MISS_BASE)
        self.assertEqual(int((table_C - table_A).unique().item()), ODD_BASE - MISS_BASE)
        flat = state.view(-1, ROW_LEN)
        if not (("state_1104" in real) or ("state_1184" in real)):
            self.assertFalse(
                torch.equal(flat[MISS_BASE:MISS_BASE + ring], flat[HIT_BASE:HIT_BASE + ring]),
                "f(slot) content must differ between miss and hit slots, else a "
                "relative-address bug cancels out (design blind spot)",
            )

        # --- run decode 17534 for A, B, C ---
        runs = {"A": (MISS_BASE, table_A), "B": (HIT_BASE, table_B), "C": (ODD_BASE, table_C)}
        op_state = {}
        ref_state = {}
        state_ok = {}
        for label, (bslot, table) in runs.items():
            op_out, op_st = _op_decode(base, state, table)
            ref_cmp, ref_mask, ref_st = _ref_decode(base, state, table)
            op_state[label] = op_st
            ref_state[label] = ref_st
            state_ok[label] = _state_maxabs(op_st.view(-1, ROW_LEN), ref_st.view(-1, ROW_LEN)) < STATE_TOL

        # --- run cmp_kv 17535 for A and B (block-136 cmp_kv) ---
        # NOTE the cmp_kv step has its own start_pos=17535, so it needs its own
        # table (positions 17527..17535) -- reusing the 17534 tables would map
        # the carry read to the wrong absolute slots.
        table_A_cmp = _build_slot_layout_table(
            [MISS_BASE], [CMP_START_POS], SEQUSED, COFF, CMP_RATIO, RING_SIZE
        )
        table_B_cmp = _build_slot_layout_table(
            [HIT_BASE], [CMP_START_POS], SEQUSED, COFF, CMP_RATIO, RING_SIZE
        )
        cmp_op = {}
        cmp_ref = {}
        cmp_mask = {}
        cmp_ok = {}
        for label, table in (("A", table_A_cmp), ("B", table_B_cmp)):
            op_out, _ = _op_decode(p_cmp, state, table)
            ref_cmp, ref_mask, _ = _ref_decode(p_cmp, state, table)
            cmp_op[label] = op_out
            cmp_ref[label] = ref_cmp
            cmp_mask[label] = ref_mask
            cmp_ok[label] = _cmp_maxabs(op_out, ref_cmp, ref_mask) < CMP_TOL

        # --- diagnosis ---
        def _r(label, base_slot):
            r34 = START_POS % RING_SIZE
            r35 = (START_POS + 1) % RING_SIZE
            o = op_state[label].view(-1, ROW_LEN)
            r = ref_state[label].view(-1, ROW_LEN)
            return (
                (o[base_slot + r34] - r[base_slot + r34]).abs().max().item(),
                (o[base_slot + r35] - r[base_slot + r35]).abs().max().item(),
            )

        op_all_identical = (
            torch.equal(op_state["A"], op_state["B"])
            and torch.equal(op_state["B"], op_state["C"])
        )

        all_state_ok = all(state_ok.values())
        all_cmp_ok = all(cmp_ok.values())

        if all_state_ok and all_cmp_ok:
            verdict = (
                "SLOT-INVARIANT -> op resolves absolute slots faithfully "
                "(matches CPU reference on miss/+80/odd-page). Hidden input, "
                "if the online divergence persists, is OUTSIDE the op"
            )
        elif op_all_identical and not all_state_ok:
            verdict = (
                "RELATIVE-ADDRESS bug -> op output is bit-identical across "
                "layouts yet wrong vs reference: it reads/writes a FIXED "
                "relative offset, ignoring the absolute slot"
            )
        elif state_ok["A"] and not (state_ok["B"] and state_ok["C"]):
            verdict = (
                "ABSOLUTE-SLOT sensitive -> miss layout matches reference but "
                "+80/+8 do not: (alpha) read / (gamma) write mis-resolves the "
                "page-boundary slot"
            )
        elif state_ok["A"] and state_ok["B"] and not state_ok["C"]:
            verdict = (
                "PAGE-PARITY bug -> even pages (miss/+80) match reference, "
                "the odd page (+8) does not"
            )
        else:
            verdict = (
                "INDETERMINATE -> did NOT capture a content-level signal (Q2 "
                "bar: must reach the online ~27, not ~1e-8)"
            )

        print(
            f"[SLOTLAYOUT] decode start_pos={START_POS} seqused={SEQUSED} "
            f"head_dim={HEAD_DIM} coff={COFF} cmp_ratio={CMP_RATIO} "
            f"ring_size={RING_SIZE} real_inputs={'YES' if env_path else 'NO'}",
            flush=True,
        )
        print(
            f"[SLOTLAYOUT] A miss={MISS_BASE} B hit={HIT_BASE} (+{HIT_BASE - MISS_BASE}) "
            f"C odd={ODD_BASE} (+{ODD_BASE - MISS_BASE}) | state content=f(absolute slot)",
            flush=True,
        )
        for label, (bslot, _) in runs.items():
            d34, d35 = _r(label, bslot)
            print(
                f"[SLOTLAYOUT] state {label} (base {bslot}): op-vs-ref maxabs "
                f"pos17534={d34:.3e} pos17535={d35:.3e} | "
                f"match(state<{STATE_TOL:g})={state_ok[label]}",
                flush=True,
            )
        for label in ("A", "B"):
            cm = _cmp_maxabs(cmp_op[label], cmp_ref[label], cmp_mask[label])
            print(
                f"[SLOTLAYOUT] cmp_kv {label} (start={CMP_START_POS} seqused=1, "
                f"block-136): op-vs-ref maxabs={cm:.3e} | "
                f"match(cmp<{CMP_TOL:g})={cmp_ok[label]}",
                flush=True,
            )
        print(
            f"[SLOTLAYOUT] op A==B==C bit-identical: {op_all_identical} "
            f"(with f(slot), True => relative addressing)",
            flush=True,
        )
        print(f"[SLOTLAYOUT-VERDICT] {verdict}", flush=True)
        print(
            "[KSTATELOC] kernel-side stateLoc diagnostic enabled "
            "(SGL_DSV4_STATELOC_DUMP=1); grep the run log with "
            "grep '[KSTATELOC]' to read op=L/R/W stateLoc/tableColumn for A vs B",
            flush=True,
        )

    def test_decode_slot_layout_batch2(self):
        """Optional batch=2 cross-check (README §226-6i): two decode requests
        in ONE op call sharing ONE state_cache, to expose any per-batch
        interaction that two independent single-batch calls cannot (the online
        path is same-batch multi-request against a shared pool). Opt-in via
        SGL_DSV4_SLOTTEST_BATCH2=1; skipped by default (it doubles the device
        run and is not part of the primary verdict)."""
        if os.environ.get(ENV_BATCH2) != "1":
            self.skipTest(f"set {ENV_BATCH2}=1 to run the optional batch=2 check")

        if not _is_arch35():
            self.skipTest("A5 (arch35) EXPLICIT state layout only")

        p = _make_inputs(
            [START_POS, START_POS],
            SEQUSED,
            COFF,
            CMP_RATIO,
            HEAD_DIM,
            HIDDEN,
            2,
            "TH",
            torch.bfloat16,
            2,
            16,
            ring_size=RING_SIZE,
            total_seq=START_POS + SEQUSED,
        )
        block_num = p["state_cache"].shape[0]
        ring = p["state_cache"].shape[1]
        gen = torch.Generator().manual_seed(SEED)
        state = _make_absolute_slot_state(block_num, ring, ROW_LEN, gen)
        table = _build_slot_layout_table(
            [MISS_BASE, HIT_BASE], [START_POS, START_POS], SEQUSED, COFF, CMP_RATIO, RING_SIZE
        )

        op_out, op_st = _op_decode(p, state, table)
        ref_cmp, ref_mask, ref_st = _ref_decode(p, state, table)

        maxabs = _state_maxabs(op_st.view(-1, ROW_LEN), ref_st.view(-1, ROW_LEN))
        print(
            f"[SLOTLAYOUT-BATCH2] shared state_cache batch=2 (miss={MISS_BASE} "
            f"hit={HIT_BASE}): op-vs-ref maxabs={maxabs:.3e} | "
            f"match(state<{STATE_TOL:g})={maxabs < STATE_TOL}",
            flush=True,
        )
        self.assertLess(maxabs, STATE_TOL, "batch=2 shared-state op diverges from reference")


if __name__ == "__main__":
    unittest.main(verbosity=2)
