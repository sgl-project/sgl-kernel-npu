"""Correctness test for the sparse_attn_sharedkv main operator.

The metadata tensor below is a deliberately minimal, single-active-core fixture.
It validates the main kernel independently and is not an implementation of the
metadata operator.
"""

import math
import unittest

import sgl_kernel_npu  # noqa: F401 - loads the torch custom-op library
import torch
import torch_npu


def _single_core_metadata(device):
    """Assign the full BN2/M/S2 range to AIC core 0."""
    metadata = torch.zeros(1024, dtype=torch.int32, device=device)
    # FA metadata is [36, 8]: enable, BN2 start, M start, S2 start,
    # BN2 end, M end, S2 end, first-FD-workspace-index.
    metadata[0] = 1  # core enable
    metadata[1] = 0  # BN2 start
    metadata[2] = 0  # M start; 0 means the beginning of the BN2 range
    metadata[3] = 0  # S2 start
    metadata[4] = 1  # BN2 end (exclusive when M/S2 end are both zero)
    metadata[5] = 0  # M end; 0 means all M blocks in the last BN2
    metadata[6] = 0  # S2 end; 0 means all S2 blocks in the last M block
    return metadata


def _page_attention_cache(cache, block_table, sequence_length):
    block_size = cache.shape[1]
    blocks = (sequence_length + block_size - 1) // block_size
    pages = [cache[int(block_table[i])].reshape(block_size, -1) for i in range(blocks)]
    return torch.cat(pages, dim=0)[:sequence_length]


def _reference_swa(
    q,
    ori_kv,
    ori_block_table,
    sequence_length,
    scale,
    ori_win_left,
    ori_win_right,
):
    # The operator uses the same shared tensor as K and V. Shape after cache
    # reconstruction is [S2, D], while q is [B=1, S1, N1, D].
    kv = _page_attention_cache(
        ori_kv.cpu().float(), ori_block_table.cpu()[0], sequence_length
    )
    query = q.cpu().float()[0]
    outputs = []
    for q_idx in range(query.shape[0]):
        aligned_kv_idx = sequence_length - query.shape[0] + q_idx
        left = max(aligned_kv_idx - ori_win_left, 0)
        right = min(aligned_kv_idx + ori_win_right, sequence_length - 1)
        visible_kv = kv[left : right + 1]
        scores = torch.matmul(query[q_idx], visible_kv.transpose(0, 1)) * scale
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1), visible_kv))
    return torch.stack(outputs, dim=0).unsqueeze(0)


class TestSparseAttnSharedkv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        torch_npu.npu.set_device(0)

    def _make_swa_bsnd_pa_nd_inputs(self, dtype, q_len=1):
        torch.manual_seed(20260812)
        device = torch.device("npu:0")
        batch, q_heads, head_dim = 1, 64, 512
        kv_len, block_size = 640, 128
        physical_blocks = 6
        scale = 1.0 / math.sqrt(head_dim)

        q = (torch.randn(batch, q_len, q_heads, head_dim) * 0.1 + 0.05).to(
            dtype=dtype, device=device
        )
        ori_kv_cpu = torch.randn(physical_blocks, block_size, 1, head_dim) * 0.5
        # The logical sequence includes physical page 5 and excludes page 2.
        # Opposite offsets on those pages make a broken PA block-table lookup
        # fail clearly. A 640-token sequence also exercises two S2 tiles.
        ori_kv_cpu[2] += 2.0
        ori_kv_cpu[5] -= 2.0
        ori_kv = ori_kv_cpu.to(dtype=dtype, device=device)
        ori_block_table = torch.tensor(
            [[4, 1, 5, 0, 3]], dtype=torch.int32, device=device
        )
        seqused_kv = torch.tensor([kv_len], dtype=torch.int32, device=device)
        # Disable the attention-sink floor so the reference is plain softmax.
        sinks = torch.full((q_heads,), -1.0e4, dtype=torch.float32, device=device)
        metadata = _single_core_metadata(device)

        return {
            "q": q,
            "ori_kv": ori_kv,
            "ori_block_table": ori_block_table,
            "seqused_kv": seqused_kv,
            "sinks": sinks,
            "metadata": metadata,
            "softmax_scale": scale,
            "cmp_ratio": 1,
            "ori_win_left": kv_len - 1,
            "ori_win_right": 0,
            "layout_q": "BSND",
            "layout_kv": "PA_ND",
        }, kv_len

    @staticmethod
    def _run_operator(inputs):
        return torch.ops.npu.sparse_attn_sharedkv(**inputs)

    def test_public_schema_matches_upstream_binding(self):
        schema = str(torch.ops.npu.sparse_attn_sharedkv.default._schema)
        self.assertNotIn("ori_kv_stride", schema)
        self.assertNotIn("cmp_kv_stride", schema)
        self.assertIn("*", schema)
        self.assertIn("float softmax_scale=0.", schema)
        self.assertIn("int cmp_ratio=0", schema)
        self.assertIn("int ori_win_left=128", schema)

    def _run_swa_bsnd_pa_nd(self, dtype):
        inputs, kv_len = self._make_swa_bsnd_pa_nd_inputs(dtype)

        actual, softmax_lse = self._run_operator(inputs)
        torch_npu.npu.synchronize()

        expected = _reference_swa(
            inputs["q"],
            inputs["ori_kv"],
            inputs["ori_block_table"],
            kv_len,
            inputs["softmax_scale"],
            inputs["ori_win_left"],
            inputs["ori_win_right"],
        )
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(softmax_lse.dtype, torch.float32)
        self.assertEqual(softmax_lse.numel(), 0)

    def test_swa_bsnd_pa_nd_fp16(self):
        self._run_swa_bsnd_pa_nd(torch.float16)

    def test_swa_bsnd_pa_nd_bf16(self):
        self._run_swa_bsnd_pa_nd(torch.bfloat16)

    def test_swa_bsnd_pa_nd_eager_prefill(self):
        inputs, kv_len = self._make_swa_bsnd_pa_nd_inputs(torch.float16, q_len=2)
        actual, _ = self._run_operator(inputs)
        torch_npu.npu.synchronize()
        expected = _reference_swa(
            inputs["q"],
            inputs["ori_kv"],
            inputs["ori_block_table"],
            kv_len,
            inputs["softmax_scale"],
            inputs["ori_win_left"],
            inputs["ori_win_right"],
        )
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=2e-2, atol=2e-2)

    def test_swa_bsnd_pa_nd_npu_graph(self):
        inputs, kv_len = self._make_swa_bsnd_pa_nd_inputs(torch.float16)

        # The ordinary operator call acts as the dummy warmup: outside capture,
        # a cache miss allocates and fills a stable tiling slot.
        self._run_operator(inputs)
        torch_npu.npu.synchronize()

        graph = torch_npu.npu.NPUGraph()
        capture_stream = torch_npu.npu.Stream()
        with torch_npu.npu.graph(
            graph, stream=capture_stream, auto_dispatch_capture=True
        ):
            graph_out, softmax_lse = self._run_operator(inputs)
        torch_npu.npu.synchronize()

        # Replays must read the current contents of the stable graph inputs.
        for seed, offset in ((20260813, -0.03), (20260814, 0.08)):
            torch.manual_seed(seed)
            inputs["q"].copy_(torch.randn_like(inputs["q"]) * 0.1 + offset)
            graph.replay()
            torch_npu.npu.synchronize()
            expected = _reference_swa(
                inputs["q"],
                inputs["ori_kv"],
                inputs["ori_block_table"],
                kv_len,
                inputs["softmax_scale"],
                inputs["ori_win_left"],
                inputs["ori_win_right"],
            )
            torch.testing.assert_close(
                graph_out.cpu().float(), expected, rtol=2e-2, atol=2e-2
            )

        self.assertEqual(softmax_lse.dtype, torch.float32)
        self.assertEqual(softmax_lse.numel(), 0)


if __name__ == "__main__":
    unittest.main()
