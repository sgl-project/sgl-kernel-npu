# Fused-interface consistency tests for torch.ops.npu.minimax_indexer:
#   1. direct mode (req_to_token + req_pool_indices) == block_table mode, bit-exact
#   2. append_local=1 (fused causal-local append, [QH,B,topk+1]) == manual emulation
#      of the triton append_local_block_to_topk_idx semantics, bit-exact
#   3. output memory layout [QH, B, ..] (view == correct data, no permute needed)
import sys
import unittest

import numpy as np
import sgl_kernel_npu
import torch
import torch_npu

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
QH, D = 64, 128


def _emulate_append(topk_idx, seq_lens, block_size, num_blocks):
    """Mirror _append_local_block_to_topk_idx_kernel semantics on CPU."""
    kvh, B, topk = topk_idx.shape
    out = np.full((kvh, B, topk + 1), -1, dtype=np.int32)
    for b in range(B):
        query_pos = max(int(seq_lens[b]) - 1, 0)
        local_blk = min(query_pos // block_size, num_blocks - 1)
        for h in range(kvh):
            cand = topk_idx[h, b].copy()
            valid = (cand >= 0) & (cand < num_blocks) & (cand * block_size <= query_pos)
            cand_out = np.where(valid, cand, -1)
            out[h, b, :topk] = cand_out
            out[h, b, topk] = -1 if (cand_out == local_blk).any() else local_blk
    return out


def _make_inputs(B, seq, block_size=128, topk=16):
    nb = (seq + block_size - 1) // block_size
    num_phys = B * nb
    rng = np.random.RandomState(7)
    q = rng.uniform(-1, 1, (B, 1, QH, D)).astype(np.float32)
    key = rng.uniform(-1, 1, (num_phys, block_size, 1, D)).astype(np.float32)
    block_table = np.zeros((B, nb), dtype=np.int32)
    for b in range(B):
        block_table[b] = np.arange(b * nb, (b + 1) * nb)
    seq_lens = np.full((B,), seq, dtype=np.int32)
    # req_to_token: [B, max_slots] row b maps token slot -> physical token id.
    # Physical block b*bs+t is stored at row b, slot b*bs+t (identity mapping).
    max_slots = nb * block_size
    req_to_token = np.full((B, max_slots), -1, dtype=np.int32)
    for b in range(B):
        req_to_token[b, : seq] = np.arange(b * nb * block_size, b * nb * block_size + seq)
    req_pool_indices = np.arange(B, dtype=np.int32)
    return q, key, block_table, seq_lens, req_to_token, req_pool_indices


class TestIndexerFusedInterface(unittest.TestCase):
    def _call(self, q_t, k_t, sl_t, bt_t, rtt_t, rpi_t, append_local):
        w_t = torch.zeros((q_t.shape[0], 1, QH), dtype=torch.bfloat16).npu()
        aq_t = torch.ones(q_t.shape[0], dtype=torch.int32).npu()
        return torch.ops.npu.minimax_indexer(
            q_t, k_t, w_t, aq_t, sl_t, bt_t, "BSND", "PA_BSND", 16, 0, 0, 0,
            1.0 / np.sqrt(D), rtt_t, rpi_t, append_local)

    def test_direct_equals_blocktable(self):
        for seq in (8192, 32768, 131072):
            B = 1
            q, key, bt, sl, rtt, rpi = _make_inputs(B, seq)
            q_t = torch.from_numpy(q).to(torch.bfloat16).npu()
            k_t = torch.from_numpy(key).to(torch.bfloat16).npu()
            sl_t = torch.from_numpy(sl).npu()
            bt_t = torch.from_numpy(bt).npu()
            rtt_t = torch.from_numpy(rtt).npu()
            rpi_t = torch.from_numpy(rpi).npu()
            out_bt = self._call(q_t, k_t, sl_t, bt_t, None, None, 0).view(QH, B, 16).cpu().numpy()
            out_direct = self._call(q_t, k_t, sl_t, None, rtt_t, rpi_t, 0).view(QH, B, 16).cpu().numpy()
            self.assertTrue(
                np.array_equal(out_bt, out_direct),
                f"direct vs block_table mismatch at seq={seq}: "
                f"{(out_bt != out_direct).sum()} elems differ",
            )
            print(f"  [direct==block_table] seq={seq}: bit-exact {np.array_equal(out_bt, out_direct)}")

    def test_append_fused_equals_emulated(self):
        for seq in (8192, 32768, 131072):
            B = 1
            q, key, bt, sl, rtt, rpi = _make_inputs(B, seq)
            q_t = torch.from_numpy(q).to(torch.bfloat16).npu()
            k_t = torch.from_numpy(key).to(torch.bfloat16).npu()
            sl_t = torch.from_numpy(sl).npu()
            bt_t = torch.from_numpy(bt).npu()
            # non-fused topk [QH, B, 16] -> emulate append on CPU
            out_topk = self._call(q_t, k_t, sl_t, bt_t, None, None, 0).view(QH, B, 16).cpu().numpy()
            num_blocks = (seq + 127) // 128
            ref = _emulate_append(out_topk, sl, 128, num_blocks)
            # fused append [QH, B, 17]
            out_fused = self._call(q_t, k_t, sl_t, bt_t, None, None, 1).view(QH, B, 17).cpu().numpy()
            self.assertTrue(
                np.array_equal(out_fused, ref),
                f"fused append vs emulated mismatch at seq={seq}: "
                f"{(out_fused != ref).sum()} elems differ",
            )
            print(f"  [append fused==emulated] seq={seq}: bit-exact {np.array_equal(out_fused, ref)}")

    def test_append_fused_direct(self):
        # direct + append combined
        seq = 65536
        B = 2
        q, key, bt, sl, rtt, rpi = _make_inputs(B, seq)
        q_t = torch.from_numpy(q).to(torch.bfloat16).npu()
        k_t = torch.from_numpy(key).to(torch.bfloat16).npu()
        sl_t = torch.from_numpy(sl).npu()
        bt_t = torch.from_numpy(bt).npu()
        rtt_t = torch.from_numpy(rtt).npu()
        rpi_t = torch.from_numpy(rpi).npu()
        out_bt = self._call(q_t, k_t, sl_t, bt_t, None, None, 1).view(QH, B, 17).cpu().numpy()
        out_direct = self._call(q_t, k_t, sl_t, None, rtt_t, rpi_t, 1).view(QH, B, 17).cpu().numpy()
        self.assertTrue(np.array_equal(out_bt, out_direct), "direct+append vs bt+append mismatch")
        print(f"  [direct+append == bt+append] seq={seq} B={B}: bit-exact")


if __name__ == "__main__":
    unittest.main(verbosity=2)
