# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import unittest

import torch
import torch_npu

from test_compressor import _is_arch35, _make_inputs

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

# Carry window = positions 17528..17533 (6 rows), bit-identical in both layouts
# (§220). The two tail slots (residues 6,7) are positions 17534/17535 -- the
# rows the decode step writes back / the divergence concentrates in.
CARRY_NROWS = 6

SEED = 20261010

# max-abs-diff threshold separating "LOGIC" (online ~2e-3 element / ~27 sum)
# from "numerical ~1e-8" (T5 K-split). Reported raw so the reader can judge.
LOGIC_MAXABS = 1e-3


def _build_slot_layout_table(start_pos, seqused, base_slot, coff, cmp_ratio, ring_size):
    """EXPLICIT state_block_table for the fixed decode call shape.

    The kernel indexes the table at column
        tableColumn = coff*cmp_ratio + pos - start_pos   (arch35
        compressor_block_vec.h ReadFromCacheState/WriteToCacheState)
    i.e. column c -> position (start_pos - history + c). Each position maps to
    its page row: slot = base_slot + (pos % ring_size), reproducing the
    production mapping state_loc = swa_page*ring_size + swa_loc % ring_size
    with swa_page = base_slot // ring_size (138 for miss, 148 for hit).
    """
    history_size = coff * cmp_ratio
    capacity = seqused
    table = torch.empty((1, history_size + capacity), dtype=torch.int32)
    for column in range(history_size + capacity):
        position = start_pos - history_size + column
        table[0, column] = base_slot + (position % ring_size)
    return table


class TestCompressorSlotLayout(unittest.TestCase):
    def test_decode_slot_layout_invariance(self):
        """Single-operator probe: does the c4 compressor depend on the absolute
        state SLOT (page boundary) when the decode call shape is held fixed?

        This is NOT T5 (test_compressor_resume_boundary_state): T5 varies the
        CALL shape (full prefill [0,17536) vs prefix+suffix [16384,17536)) and
        only reproduces the ~1e-8 numerical (K-split) phenomenon -- the wrong
        one (§209-§213). Here the call shape is FIXED to the decode step that
        diverges online (start_pos=17534, seqused=1) and the ONLY variable is
        where the identical carry content sits in state_cache + the
        state_block_table it points at:
          Run A (miss): carry at flat slots 1104..1111.
          Run B (hit):  the SAME carry at 1184..1191 (+80 = +10 SWA pages).
        Everything else (x, wkv/wgate, ape, norm_weight, rope, cu_seqlens,
        seqused, start_pos) is shared bit-for-bit.

        Verdict semantics (printed, not asserted -- the outcome is the answer):
          * differ AND ~LOGIC level (max-abs-diff >> 1e-8)  => (alpha) the op
            is sensitive to the slot's absolute position at the page boundary.
          * identical                                        => the op is
            slot-invariant => the hidden input is OUTSIDE the op
            (pool/framework/gamma); do NOT change the op, return to the
            mechanism layer.
          * only ~1e-8 difference                             => did NOT capture
            the hidden input (Q2 bar); the online divergence is ~27 on the row
            sum, so a ~1e-8 delta must be flagged as insufficient.

        NOTE on cmp_kv: a single non-boundary decode step (17534 % 4 == 2)
        completes no cmp_ratio group (compressTcSize==0), so the host returns an
        at::empty output tensor (uninitialized). Its bit-identity is therefore
        NOT meaningful evidence; the meaningful signal is the written-back state
        at the 17534/17535 slots. Both are printed for completeness.

        KSTATELOC: the test sets SGL_DSV4_STATELOC_DUMP=1 at import; capture the
        run and `grep '\\[KSTATELOC\\]' <log>` to read the kernel's actual
        stateLoc/tableColumn for run A vs run B (op=L read-left, op=R read,
        op=W write).

        Op call shape + _make_inputs usage mirrored from test_compressor.py:
        the torch.ops.npu.compressor(...) argument order/attrs are copied from
        test_compressor_resume_boundary_state (test_compressor.py:1089-1110) and
        _run_case (test_compressor.py:708-728); head_dim=128/hidden=1024 mirror
        that same test (test_compressor.py:1051). Only the state cache content
        and the state_block_table are replaced with the slot-layout A/B forms.
        """
        if not _is_arch35():
            self.skipTest("A5 (arch35) EXPLICIT state layout only")

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

        block_num = base["state_cache"].shape[0]
        ring = base["state_cache"].shape[1]
        self.assertEqual(ring, RING_SIZE)

        # Carry + output-slot initial content, bit-identical across A and B.
        # The exact online 512-element rows are NOT available (swanew_10_10_1.txt
        # carries only per-row aggregates sum/absmax/v0/md5, e.g. sums
        # -20.27, -58.79, +4.03, -124.50, -11.30, -97.89 for positions
        # 17528..17533), so the rows are synthesized with a fixed seed. The
        # probe only needs them IDENTICAL across the two runs.
        gen = torch.Generator().manual_seed(SEED)
        carry = (torch.randn(CARRY_NROWS, ROW_LEN, generator=gen) * 0.01).float()
        out_init = (torch.randn(RING_SIZE - CARRY_NROWS, ROW_LEN, generator=gen) * 0.01).float()

        def build_state(base_slot):
            state = torch.zeros(block_num, ring, ROW_LEN)
            flat = state.view(-1, ROW_LEN)
            flat[base_slot : base_slot + CARRY_NROWS] = carry
            flat[base_slot + CARRY_NROWS : base_slot + ring] = out_init
            return state

        state_A = build_state(MISS_BASE)
        state_B = build_state(HIT_BASE)
        table_A = _build_slot_layout_table(
            START_POS, SEQUSED, MISS_BASE, COFF, CMP_RATIO, RING_SIZE
        )
        table_B = _build_slot_layout_table(
            START_POS, SEQUSED, HIT_BASE, COFF, CMP_RATIO, RING_SIZE
        )

        # Sanity: the two runs differ ONLY in where the identical carry/output
        # rows sit, and in the table (+80) that points at them.
        flat_A = state_A.view(-1, ROW_LEN)
        flat_B = state_B.view(-1, ROW_LEN)
        self.assertTrue(
            torch.equal(
                flat_A[MISS_BASE : MISS_BASE + ring],
                flat_B[HIT_BASE : HIT_BASE + ring],
            ),
            "carry/output rows must be bit-identical across the two layouts",
        )
        self.assertEqual(
            int((table_B - table_A).unique().item()), HIT_BASE - MISS_BASE
        )

        def run(state_cpu, table):
            state_npu = state_cpu.clone().npu()
            out = torch.ops.npu.compressor(
                base["x"].npu(),
                base["wkv"].npu(),
                base["wgate"].npu(),
                state_npu,
                base["ape"].npu(),
                base["norm_weight"].npu(),
                base["rope_sin"].npu(),
                base["rope_cos"].npu(),
                state_block_table=table.npu(),
                cu_seqlens=base["cu_seqlens"].npu(),
                seqused=torch.tensor(base["seqused"], dtype=torch.int32).npu(),
                start_pos=torch.tensor(base["start_pos"], dtype=torch.int32).npu(),
                rope_head_dim=64,
                cmp_ratio=CMP_RATIO,
                coff=COFF,
                norm_eps=1e-6,
                rotary_mode=2,
                cache_mode=2,
                state_cache_stride_dim0=0,
            )
            torch_npu.npu.synchronize()
            return out.cpu(), state_npu.cpu()

        out_A, state_A_out = run(state_A, table_A)
        out_B, state_B_out = run(state_B, table_B)

        flat_A_out = state_A_out.view(-1, ROW_LEN)
        flat_B_out = state_B_out.view(-1, ROW_LEN)

        r34 = START_POS % RING_SIZE
        r35 = (START_POS + 1) % RING_SIZE
        row34_A = flat_A_out[MISS_BASE + r34]
        row34_B = flat_B_out[HIT_BASE + r34]
        row35_A = flat_A_out[MISS_BASE + r35]
        row35_B = flat_B_out[HIT_BASE + r35]

        # cmp_kv: uninitialized for a single non-boundary decode -- reported for
        # completeness, not a meaningful signal.
        cmp_shape_eq = out_A.shape == out_B.shape
        cmp_bit = bool(cmp_shape_eq and torch.equal(out_A, out_B))
        cmp_maxabs = (
            float((out_A - out_B).abs().max())
            if cmp_shape_eq and out_A.numel() > 0
            else float("nan")
        )

        s34_bit = bool(torch.equal(row34_A, row34_B))
        s34_maxabs = float((row34_A - row34_B).abs().max())
        s34_dsum = float((row34_A - row34_B).sum())
        s35_bit = bool(torch.equal(row35_A, row35_B))
        s35_maxabs = float((row35_A - row35_B).abs().max())
        s35_dsum = float((row35_A - row35_B).sum())

        worst_maxabs = max(s34_maxabs, s35_maxabs)
        if s34_bit and s35_bit:
            verdict = (
                "IDENTICAL -> op is SLOT-INVARIANT -> hidden input is OUTSIDE "
                "the op (pool/framework/gamma); do NOT change the op"
            )
        elif worst_maxabs >= LOGIC_MAXABS:
            verdict = (
                "DIFFER (LOGIC level) -> alpha CONFIRMED: op is sensitive to "
                "the slot's absolute position at the page boundary"
            )
        else:
            verdict = (
                "DIFFER (~numerical only) -> did NOT capture the hidden input "
                "(Q2 bar: must reach the online ~27, not ~1e-8)"
            )

        print(
            f"[SLOTLAYOUT] decode start_pos={START_POS} seqused={SEQUSED} "
            f"head_dim={HEAD_DIM} coff={COFF} cmp_ratio={CMP_RATIO} "
            f"ring_size={RING_SIZE}",
            flush=True,
        )
        print(
            f"[SLOTLAYOUT] carry slots miss {MISS_BASE}..{MISS_BASE + ring - 1} "
            f"(SWA page {MISS_BASE // RING_SIZE}) vs hit "
            f"{HIT_BASE}..{HIT_BASE + ring - 1} (SWA page {HIT_BASE // RING_SIZE}, "
            f"+{HIT_BASE - MISS_BASE})",
            flush=True,
        )
        print(
            f"[SLOTLAYOUT] cmp_kv  shape A={tuple(out_A.shape)} B={tuple(out_B.shape)} "
            f"| bit-identical={cmp_bit} | maxabs={cmp_maxabs:.3e}  "
            f"(NOTE: uninitialized for single non-boundary decode)",
            flush=True,
        )
        print(
            f"[SLOTLAYOUT] state pos={START_POS} (slot {MISS_BASE + r34} vs "
            f"{HIT_BASE + r34}): bit-identical={s34_bit} | "
            f"maxabs={s34_maxabs:.3e} | dsum={s34_dsum:.3e}",
            flush=True,
        )
        print(
            f"[SLOTLAYOUT] state pos={START_POS + 1} (slot {MISS_BASE + r35} vs "
            f"{HIT_BASE + r35}): bit-identical={s35_bit} | "
            f"maxabs={s35_maxabs:.3e} | dsum={s35_dsum:.3e}",
            flush=True,
        )
        print(f"[SLOTLAYOUT-VERDICT] {verdict}", flush=True)
        print(
            "[KSTATELOC] kernel-side stateLoc diagnostic enabled "
            "(SGL_DSV4_STATELOC_DUMP=1); grep the run log with "
            "grep '[KSTATELOC]' to read op=L/R/W stateLoc/tableColumn for A vs B",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
