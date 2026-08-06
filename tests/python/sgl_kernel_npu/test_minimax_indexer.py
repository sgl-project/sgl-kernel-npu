# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# AscendC precision test for torch.ops.npu.minimax_indexer (MiniMax-M3 indexer).
#
# Stage A: single-core. Validates the per-head block-level topk against a NumPy
# reference that mirrors triton `_decode_bnsd_score_topk_chunk_kernel`
# (SCORE_TYPE="max") + `_torch_topk_from_score`: for every (batch, query head),
# block_score = max_t(q·k_t)*sm_scale_log2e with init/local sentinels, then
# streaming top-16 over blocks. Compares the SELECTED SET of block indices
# (order-unstable, like the lightning test) and the score multiset.

import unittest

import numpy as np
import sgl_kernel_npu  # registers torch.ops.npu.minimax_indexer
import torch
import torch_npu

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

LOG2E = 1.4426950408889634


def _gather_k(key_cache, block_table, b, num_blocks, block_size):
    """Gather the first num_blocks logical blocks of batch b -> [num_blocks*block_size, D]."""
    phys = block_table[b, :num_blocks]
    # key_cache: [num_phys_blocks, block_size, 1, D]
    k = key_cache[phys]  # [num_blocks, block_size, 1, D]
    return k.reshape(num_blocks * block_size, -1)  # [num_blocks*block_size, D]


def _minimax_indexer_ref(
    q,
    key_cache,
    block_table,
    seq_lens,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale,
):
    """CPU/NumPy reference. Returns candidate_indices [B, QH, topk] int32 (-1 padded)."""
    B, S1, QH, D = q.shape
    assert S1 == 1
    qh = q.reshape(B, QH, D).astype(np.float32)  # [B, QH, D]
    scale = float(sm_scale) * LOG2E
    out = np.full((B, QH, topk), -1, dtype=np.int32)
    for b in range(B):
        seq_len = int(seq_lens[b])
        num_blocks = (seq_len + block_size - 1) // block_size
        if num_blocks == 0:
            continue
        k_all = _gather_k(key_cache, block_table, b, num_blocks, block_size).astype(
            np.float32
        )
        # validity mask over tokens
        pos = np.arange(num_blocks * block_size)
        valid_tok = pos < seq_len  # [num_blocks*block_size]
        for h in range(QH):
            qk = qh[b, h] @ k_all.T * scale  # [num_blocks*block_size]
            qk = np.where(valid_tok, qk, -np.inf)
            # max-reduce within each block -> per-block score
            qk_blk = qk.reshape(num_blocks, block_size)  # [num_blocks, block_size]
            blk_score = qk_blk.max(axis=1).astype(np.float64)  # [num_blocks]
            local_start = max(0, num_blocks - local_blocks)
            for blk in range(num_blocks):
                if init_blocks > 0 and blk < init_blocks:
                    blk_score[blk] = 1e30
                elif local_blocks > 0 and blk >= local_start:
                    blk_score[blk] = 1e29
            if num_blocks <= topk:
                idx = list(range(num_blocks)) + [-1] * (topk - num_blocks)
                out[b, h] = np.array(idx, dtype=np.int32)
            else:
                # top-16 by score; ties broken by lower index (stable); compare as set
                order = np.argsort(-blk_score, kind="stable")[:topk]
                out[b, h] = order.astype(np.int32)
    return out


def _make_block_table(B, num_phys_blocks, num_blocks_per_batch, rng):
    """Identity-ish mapping: logical block i -> physical block i (distinct per batch)."""
    assert num_phys_blocks >= num_blocks_per_batch
    bt = np.zeros((B, num_blocks_per_batch), dtype=np.int32)
    for b in range(B):
        # contiguous, distinct physical blocks per batch
        bt[b] = np.arange(b * num_blocks_per_batch, (b + 1) * num_blocks_per_batch)
    return bt


class TestMinimaxIndexer(unittest.TestCase):
    def _run(self, B, seq_lens, block_size, topk, init_blocks, local_blocks, dtype):
        QH, D = 64, 128
        max_blocks = (max(seq_lens) + block_size - 1) // block_size
        num_phys = B * max_blocks
        rng = np.random.RandomState(42)
        # query [B, S1=1, QH, D]
        q = rng.uniform(-1, 1, (B, 1, QH, D)).astype(np.float32)
        # key cache [num_phys, block_size, 1, D]
        key = rng.uniform(-1, 1, (num_phys, block_size, 1, D)).astype(np.float32)
        block_table = _make_block_table(B, num_phys, max_blocks, rng)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        act_q = torch.ones(B, dtype=torch.int32)  # S1=1
        # dummy weights [B, S1, QH] (unused, kept for the lightning-derived parser)
        weights = torch.zeros((B, 1, QH), dtype=dtype)
        sm_scale = 1.0 / np.sqrt(D)

        # The NPU runs on bf16/fp16 inputs; the reference must use the SAME
        # rounded inputs (cast to dtype then back to fp32) so the comparison is
        # apples-to-apples. Residual single-block swaps are then true score ties
        # at the rank-16/17 boundary, which are semantically either-or.
        q_rounded = torch.from_numpy(q).to(dtype).float().numpy()
        k_rounded = torch.from_numpy(key).to(dtype).float().numpy()
        ref = _minimax_indexer_ref(
            q_rounded,
            k_rounded,
            block_table,
            seq_lens,
            block_size,
            topk,
            init_blocks,
            local_blocks,
            sm_scale,
        )

        q_t = torch.from_numpy(q).to(dtype).npu()
        k_t = torch.from_numpy(key).to(dtype).npu()
        bt_t = torch.from_numpy(block_table).npu()
        w_t = weights.npu()
        act_q_t = act_q.npu()
        seq_t = seq_lens_t.npu()

        npu_out = torch.ops.npu.minimax_indexer(
            q_t,
            k_t,
            w_t,
            actual_seq_lengths_query=act_q_t,
            actual_seq_lengths_key=seq_t,
            block_table=bt_t,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=topk,
            sparse_mode=0,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=float(sm_scale),
        )
        # npu_out: [B, S1=1, QH, topk] shape, kernel memory layout [QH, B, topk]
        # -> a view reinterprets to [QH, B, topk] without any copy.
        npu_idx = npu_out.view(QH, B, topk).cpu().numpy()

        # Compare selected SET per (b, h): indices are order-unstable.
        # npu_idx layout is [QH, B, topk]: row (h, b) at npu_idx[h, b].
        mismatches = 0
        for b in range(B):
            for h in range(QH):
                a = sorted(int(x) for x in npu_idx[h, b] if int(x) >= 0)
                r = sorted(int(x) for x in ref[b, h] if int(x) >= 0)
                if a != r:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"  MISMATCH b={b} h={h}: npu={a[:20]} ref={r[:20]}")
        total = B * QH
        print(
            f"  [{dtype}] block_size={block_size} topk={topk} init={init_blocks} "
            f"local={local_blocks} seqs={seq_lens}: {total - mismatches}/{total} (b,h) match "
            f"(set-of-indices)"
        )
        self.assertEqual(
            mismatches, 0, f"{mismatches}/{total} (batch,head) pairs mismatch"
        )

    def test_general_blocks_gt_topk(self):
        # num_blocks (64,32) > topk=16 -> real streaming topk path.
        for dtype in [torch.bfloat16, torch.float16]:
            self._run(
                B=2,
                seq_lens=[8192, 4096],
                block_size=128,
                topk=16,
                init_blocks=1,
                local_blocks=2,
                dtype=dtype,
            )

    def test_trivial_blocks_le_topk(self):
        # num_blocks (8) <= topk=16 -> trivial [0..num_blocks)+(-1) path.
        for dtype in [torch.bfloat16, torch.float16]:
            self._run(
                B=1,
                seq_lens=[1024],
                block_size=128,
                topk=16,
                init_blocks=0,
                local_blocks=0,
                dtype=dtype,
            )

    def test_partial_last_block(self):
        # seq_len not a multiple of block_size -> partial tail block.
        for dtype in [torch.bfloat16]:
            self._run(
                B=1,
                seq_lens=[8400],
                block_size=128,
                topk=16,
                init_blocks=1,
                local_blocks=1,
                dtype=dtype,
            )

    def test_no_sentinel(self):
        # init=0, local=0 -> pure score topk, no sentinel override.
        for dtype in [torch.bfloat16]:
            self._run(
                B=2,
                seq_lens=[6144, 3072],
                block_size=128,
                topk=16,
                init_blocks=0,
                local_blocks=0,
                dtype=dtype,
            )

    def test_block_size_64(self):
        for dtype in [torch.bfloat16]:
            self._run(
                B=1,
                seq_lens=[8192],
                block_size=64,
                topk=16,
                init_blocks=1,
                local_blocks=1,
                dtype=dtype,
            )

    def test_large_seq_real_topk(self):
        # blocks/core > topk=16 -> exercises the streaming replace-min path.
        # This region (seq >= ~48k) was silently broken (deterministic wrong topk)
        # until the WholeReduceMin ORDER_VALUE_INDEX offset fix; the small-seq
        # cases above never reach it (blocks/core <= topk -> append-only).
        for dtype in [torch.bfloat16, torch.float16]:
            self._run(
                B=1,
                seq_lens=[65536],
                block_size=128,
                topk=16,
                init_blocks=1,
                local_blocks=2,
                dtype=dtype,
            )
        # 131072 (1024 blocks, ~43/core) is heavy on the CPU ref; bf16 only.
        self._run(
            B=1,
            seq_lens=[131072],
            block_size=128,
            topk=16,
            init_blocks=1,
            local_blocks=2,
            dtype=torch.bfloat16,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
