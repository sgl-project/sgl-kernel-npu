"""Numerical correctness tests for quantized sparse shared-KV attention.

Single-core fixtures isolate the main kernel, while the multi-batch case uses
the real metadata builder. The reference reconstructs the dequantized KV cache
on the CPU and computes attention with PyTorch.
"""

import math
import unittest

import sgl_kernel_npu  # noqa: F401 - registers the custom operators
import torch
import torch_npu


def _single_core_metadata(device):
    """Assign the entire batch/query/KV range to cube core 0."""
    metadata = torch.zeros(1024, dtype=torch.int32, device=device)
    # The A5 FA metadata has 36 slots of 9 integers. A zero M/S2 end means
    # the end of the final BN2 range, as in test_sparse_attn_sharedkv.py.
    metadata[0] = 1  # enable core 0
    metadata[4] = 1  # exclusive BN2 end
    return metadata


def _packed_kv_and_reference(block_count, block_size):
    """Make valid 640-byte rows and the corresponding dequantized [slot, D]."""
    torch.manual_seed(20261015)
    block_bias = torch.arange(block_count, dtype=torch.float32)[:, None, None] * 0.05
    rope = (torch.randn(block_count, block_size, 64) * 0.5 + block_bias).to(
        torch.bfloat16
    )
    nope = (torch.randn(block_count, block_size, 448) * 0.5 + block_bias).to(
        torch.float8_e4m3fn
    )

    packed = torch.zeros((block_count, block_size, 1, 640), dtype=torch.uint8)
    # Each row stores 64 BF16 RoPE values, 448 FP8 NoPE values, then seven
    # E8M0 scales. Exponents 126, 127, 128 encode 0.5, 1.0, 2.0.
    scale_codes = torch.tensor(
        [126, 127, 128, 126, 127, 128, 127], dtype=torch.uint8
    )
    scale_values = torch.tensor([0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 1.0])
    packed[:, :, 0, :128] = rope.contiguous().view(torch.uint8)
    packed[:, :, 0, 128:576] = nope.contiguous().view(torch.uint8)
    packed[:, :, 0, 576:583] = scale_codes

    # The device rearranges the row to [NoPE(448), RoPE(64)] and rounds the
    # dequantized FP8 values to BF16 before the attention matmul.
    dequant_nope = nope.float().reshape(block_count, block_size, 7, 64)
    dequant_nope = (dequant_nope * scale_values[None, None, :, None]).to(
        torch.bfloat16
    )
    dequant_nope = dequant_nope.reshape(block_count, block_size, 448)
    dequantized = torch.cat((dequant_nope, rope), dim=-1)
    return packed.view(torch.float8_e4m3fn), dequantized.reshape(-1, 512)


def _reference_sparse_attention(q, kv, physical_slots, scale):
    selected_kv = kv[physical_slots.long()].float()
    scores = torch.matmul(q[0].float(), selected_kv.T) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), selected_kv).unsqueeze(0)


def _reference_paged_swa(
    q, kv, block_table, kv_length, block_size, scale, ori_win_left, ori_win_right
):
    """Apply the logical window, then resolve rows through the page table."""
    q_length = q.shape[1]
    outputs = []
    for query_idx in range(q_length):
        aligned_kv_idx = kv_length - q_length + query_idx
        left = max(aligned_kv_idx - ori_win_left, 0)
        right = min(aligned_kv_idx + ori_win_right, kv_length - 1)
        logical_slots = torch.arange(left, right + 1)
        physical_slots = (
            block_table[0, logical_slots // block_size].long() * block_size
            + logical_slots % block_size
        )
        selected_kv = kv[physical_slots].float()
        scores = torch.matmul(q[0, query_idx].float(), selected_kv.T) * scale
        outputs.append(torch.matmul(torch.softmax(scores, dim=-1), selected_kv))
    return torch.stack(outputs, dim=0).unsqueeze(0)


class TestKvQuantSparseAttnSharedkv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        if "950" not in torch_npu.npu.get_device_name(0).lower():
            raise unittest.SkipTest("the quantized kernel requires Ascend 950")
        torch_npu.npu.set_device(0)

    def test_public_schema(self):
        schema = torch.ops.npu.kv_quant_sparse_attn_sharedkv.default._schema
        arguments = {argument.name: argument for argument in schema.arguments}

        self.assertEqual(schema.arguments[0].name, "q")
        self.assertFalse(schema.arguments[0].kwarg_only)
        self.assertEqual(len(schema.returns), 2)
        for name in (
            "ori_kv",
            "cmp_kv",
            "ori_sparse_indices",
            "cmp_sparse_indices",
            "ori_block_table",
            "cmp_block_table",
            "metadata",
        ):
            with self.subTest(argument=name):
                self.assertIn(name, arguments)
                self.assertTrue(arguments[name].kwarg_only)
                self.assertIsNone(arguments[name].default_value)

        for name, default in (
            ("kv_quant_mode", 1),
            ("tile_size", 64),
            ("rope_head_dim", 64),
            ("softmax_scale", 1.0),
            ("cmp_ratio", 1),
            ("ori_win_left", 127),
            ("ori_win_right", 0),
            ("layout_q", "BSND"),
            ("layout_kv", "PA_ND"),
            ("return_softmax_lse", False),
        ):
            with self.subTest(argument=name):
                self.assertIn(name, arguments)
                self.assertTrue(arguments[name].kwarg_only)
                self.assertEqual(arguments[name].default_value, default)

    def test_rejects_fp16_query(self):
        q = torch.zeros((1, 1, 64, 512), dtype=torch.float16, device="npu:0")
        with self.assertRaisesRegex(RuntimeError, "q must be bfloat16"):
            torch.ops.npu.kv_quant_sparse_attn_sharedkv(q)

    def _make_paged_swa_inputs(self, q_length, ori_win_left):
        device = torch.device("npu:0")
        block_count, block_size, kv_length = 20, 16, 256
        scale = 1.0 / math.sqrt(512)
        packed_cpu, kv_reference = _packed_kv_and_reference(block_count, block_size)
        # All logical pages are valid, but none uses its own page number.
        block_table = torch.arange(16, dtype=torch.int32).roll(5).unsqueeze(0)
        torch.manual_seed(20261019 + q_length)
        q_cpu = (torch.randn(1, q_length, 64, 512) * 0.1 + 0.05).to(torch.bfloat16)
        kv_length_cpu = torch.tensor([kv_length], dtype=torch.int32)
        metadata = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            layout_q="BSND",
            layout_kv="PA_ND",
            seqused_kv=kv_length_cpu,
            batch_size=1,
            max_seqlen_q=q_length,
            max_seqlen_kv=kv_length,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=ori_win_left,
            ori_win_right=0,
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        inputs = {
            "q": q_cpu.to(device),
            "ori_kv": packed_cpu.to(device),
            "ori_block_table": block_table.to(device),
            "seqused_kv": kv_length_cpu.to(device),
            "sinks": torch.full((64,), -1.0e4, dtype=torch.float32, device=device),
            "metadata": metadata,
            "softmax_scale": scale,
            "cmp_ratio": 1,
            "ori_win_left": ori_win_left,
            "ori_win_right": 0,
            "layout_q": "BSND",
            "layout_kv": "PA_ND",
        }
        return inputs, q_cpu, kv_reference, block_table, kv_length, block_size

    def _assert_paged_swa(self, q_length, ori_win_left):
        inputs, q_cpu, kv_reference, block_table, kv_length, block_size = (
            self._make_paged_swa_inputs(q_length, ori_win_left)
        )
        actual, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)
        torch_npu.npu.synchronize()
        expected = _reference_paged_swa(
            q_cpu,
            kv_reference,
            block_table,
            kv_length,
            block_size,
            inputs["softmax_scale"],
            ori_win_left,
            inputs["ori_win_right"],
        )
        self.assertEqual(actual.shape, q_cpu.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(lse.numel(), 0)

    def test_swa_bsnd_pa_nd_bf16(self):
        self._assert_paged_swa(q_length=1, ori_win_left=127)

    def test_swa_bsnd_pa_nd_eager_prefill(self):
        # A zero-left window makes the two query tokens read distinct KV rows.
        self._assert_paged_swa(q_length=2, ori_win_left=0)

    def test_swa_bsnd_pa_nd_npu_graph(self):
        inputs, _, kv_reference, block_table, kv_length, block_size = (
            self._make_paged_swa_inputs(q_length=1, ori_win_left=127)
        )
        # Warm up tiling outside capture, then keep every graph input alive.
        torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)
        torch_npu.npu.synchronize()

        graph = torch_npu.npu.NPUGraph()
        capture_stream = torch_npu.npu.Stream()
        with torch_npu.npu.graph(
            graph, stream=capture_stream, auto_dispatch_capture=True
        ):
            graph_out, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)
        torch_npu.npu.synchronize()

        # Make the two replay queries point toward and away from a visible KV
        # row so stale graph inputs produce a clearly different result.
        window_start = kv_length - 1 - inputs["ori_win_left"]
        probe_slot = (
            int(block_table[0, window_start // block_size]) * block_size
            + window_start % block_size
        )
        probe_q = kv_reference[probe_slot].to("npu:0").view(1, 1, 1, 512)
        expected_outputs = []
        for sign in (-2.0, 2.0):
            inputs["q"].copy_((probe_q * sign).expand_as(inputs["q"]))
            graph.replay()
            torch_npu.npu.synchronize()
            expected = _reference_paged_swa(
                inputs["q"].cpu(),
                kv_reference,
                block_table,
                kv_length,
                block_size,
                inputs["softmax_scale"],
                inputs["ori_win_left"],
                inputs["ori_win_right"],
            )
            torch.testing.assert_close(
                graph_out.cpu().float(), expected, rtol=2e-2, atol=2e-2
            )
            expected_outputs.append(expected)
        self.assertGreater(
            (expected_outputs[0] - expected_outputs[1]).abs().max().item(), 0.05
        )
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(lse.numel(), 0)

    def _run_sparse_case(self, physical_slots, block_count):
        device = torch.device("npu:0")
        block_size = 16
        kv_length = block_count * block_size
        scale = 1.0 / math.sqrt(512)
        packed_cpu, kv_reference = _packed_kv_and_reference(block_count, block_size)

        torch.manual_seed(20261016)
        q_cpu = (torch.randn(1, 64, 512) * 0.1 + 0.05).to(torch.bfloat16)
        sparse_indices = torch.full((1, 1, 512), -1, dtype=torch.int32)
        sparse_indices[0, 0, : physical_slots.numel()] = physical_slots
        # A valid slot after the first -1 must not extend the attention range.
        sparse_indices[0, 0, physical_slots.numel() + 1] = kv_length - 1

        inputs = {
            "q": q_cpu.to(device),
            "ori_kv": packed_cpu.to(device),
            "ori_sparse_indices": sparse_indices.to(device),
            # A non-identity table catches accidental logical-page lookups.
            # Sparse indices themselves already contain physical slot IDs.
            "ori_block_table": torch.arange(block_count, dtype=torch.int32)
            .roll(3)
            .unsqueeze(0)
            .to(device),
            "cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "seqused_q": torch.tensor([1], dtype=torch.int32, device=device),
            "seqused_kv": torch.tensor([kv_length], dtype=torch.int32, device=device),
            "sinks": torch.full((64,), -1.0e4, dtype=torch.float32, device=device),
            "metadata": _single_core_metadata(device),
            "softmax_scale": scale,
            "cmp_ratio": 1,
            "ori_win_left": kv_length - 1,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        }

        actual, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(**inputs)
        torch_npu.npu.synchronize()
        expected = _reference_sparse_attention(
            q_cpu, kv_reference, physical_slots, scale
        )

        self.assertEqual(actual.shape, q_cpu.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        actual_cpu = actual.cpu().float()
        self.assertGreater(expected.abs().max().item(), 0.05)
        self.assertGreater(actual_cpu.abs().max().item(), 0.05)
        torch.testing.assert_close(actual_cpu, expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(lse.numel(), 0)

    def test_tnd_sparse_physical_slots(self):
        # The selected slots live in four different physical blocks; they are
        # neither the first logical tokens nor a contiguous slice of the cache.
        physical_slots = torch.tensor([2, 19, 35, 52], dtype=torch.int32)
        self._run_sparse_case(physical_slots, block_count=4)

    def test_tnd_sparse_indices_cross_128_tile_boundary(self):
        # More than 128 selected rows exercises two S2 tiles and their merge.
        torch.manual_seed(20261017)
        physical_slots = torch.randperm(256)[:129].sort().values.to(torch.int32)
        self._run_sparse_case(physical_slots, block_count=16)

    def test_tnd_multi_batch_128_heads_with_generated_metadata(self):
        device = torch.device("npu:0")
        block_size, block_count = 16, 24
        kv_lengths_cpu = torch.tensor([256, 320], dtype=torch.int32)
        cu_seqlens_q_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)
        scale = 1.0 / math.sqrt(512)
        packed_cpu, kv_reference = _packed_kv_and_reference(block_count, block_size)

        torch.manual_seed(20261018)
        q_cpu = (torch.randn(2, 128, 512) * 0.1 + 0.05).to(torch.bfloat16)
        physical_slots = [
            torch.randperm(256)[:129].sort().values.to(torch.int32),
            (torch.randperm(320)[:137] + 64).sort().values.to(torch.int32),
        ]
        sparse_indices = torch.full((2, 1, 512), -1, dtype=torch.int32)
        for batch_idx, slots in enumerate(physical_slots):
            sparse_indices[batch_idx, 0, : slots.numel()] = slots

        metadata = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(
            num_heads_q=128,
            num_heads_kv=1,
            head_dim=512,
            layout_q="TND",
            layout_kv="PA_ND",
            cu_seqlens_q=cu_seqlens_q_cpu,
            seqused_kv=kv_lengths_cpu,
            batch_size=2,
            max_seqlen_q=1,
            max_seqlen_kv=320,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=319,
            ori_win_right=0,
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        self.assertEqual(metadata.device.type, "npu")
        self.assertEqual(metadata.shape, (1024,))

        # Sparse indices are physical slots; deliberately use non-identity
        # block tables so a mistaken page-table lookup changes the result.
        logical_blocks = 320 // block_size
        block_ids = torch.arange(logical_blocks, dtype=torch.int32)
        block_tables = torch.stack((block_ids.roll(3), block_ids.roll(7)))
        actual, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(
            q_cpu.to(device),
            ori_kv=packed_cpu.to(device),
            ori_sparse_indices=sparse_indices.to(device),
            ori_block_table=block_tables.to(device),
            cu_seqlens_q=cu_seqlens_q_cpu.to(device),
            seqused_kv=kv_lengths_cpu.to(device),
            sinks=torch.full((128,), -1.0e4, dtype=torch.float32, device=device),
            metadata=metadata,
            softmax_scale=scale,
            cmp_ratio=1,
            ori_win_left=319,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        torch_npu.npu.synchronize()

        expected = torch.cat(
            [
                _reference_sparse_attention(
                    q_cpu[batch_idx : batch_idx + 1], kv_reference, slots, scale
                )
                for batch_idx, slots in enumerate(physical_slots)
            ]
        )
        self.assertEqual(actual.shape, q_cpu.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(lse.numel(), 0)


if __name__ == "__main__":
    unittest.main()
