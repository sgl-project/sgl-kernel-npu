"""QSA main attention: CPU reference, exact selection, and queued/graph lifetimes."""

import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.attention.qsa_prefill import qsa_prefill


@unittest.skipUnless(
    hasattr(torch.ops.npu, "qsa_prefill_runs"), "A3 QSA operator required"
)
class TestQsaPrefill(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9302026)
        torch.set_num_threads(8)

    def inputs(self, rows, length, mode, heads=16, kv_heads=2):
        capacity = max(6000, length)
        q = torch.randn(rows, heads, 256, dtype=torch.bfloat16)
        k = torch.randn(capacity, kv_heads, 256, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        table = torch.arange(length, dtype=torch.int32)[None]
        blocks = torch.arange(512, dtype=torch.int32).repeat(rows, 1)
        if mode in ("pairs", "quartets", "octets", "different"):
            step = {"pairs": 2, "quartets": 4, "octets": 8, "different": 1}[mode]
            blocks += (torch.arange(rows, dtype=torch.int32) // step)[:, None]
        elif mode == "interior":
            blocks[14::16, 271] += 7
        elif mode == "empty":
            blocks.fill_(-1)
        elif mode == "fragmented":
            blocks = torch.randint(-3, 1500, (rows, 512), dtype=torch.int32)
            blocks[:, 10] = 2147483647
            blocks[:, 20] = blocks[:, 19]
            if rows >= 16:
                blocks[:16] = blocks[0].clone()
            table = torch.randperm(capacity).int()[:length][None]
            table[:, ::13] = -1
            table[:, ::31] = capacity + 1
        return q, k, v, blocks, table

    def reference(self, q, k, v, blocks, table, length, base, scale):
        outputs = []
        for row in range(q.shape[0]):
            positions = [
                4 * block + j
                for block in blocks[row].tolist()
                if block >= 0
                for j in range(4)
                if 4 * block + j < length
            ]
            visible = base + row + 1
            positions += [
                j for j in range(visible - visible % 4, visible) if 0 <= j < length
            ]
            slots = [
                int(table[0, j])
                for j in positions
                if 0 <= int(table[0, j]) < k.shape[0]
            ]
            if slots:
                keys = k[slots].float().repeat_interleave(q.shape[1] // k.shape[1], 1)
                values = v[slots].float().repeat_interleave(q.shape[1] // k.shape[1], 1)
                logits = torch.einsum("hd,shd->hs", q[row].float(), keys) * scale
                outputs.append(torch.einsum("hs,shd->hd", logits.softmax(-1), values))
            else:
                outputs.append(torch.zeros(q.shape[1], 256))
        return torch.stack(outputs) if outputs else torch.empty(0, q.shape[1], 256)

    def test_reference_and_selection_boundaries(self):
        cases = [
            (r, 5000, 5000 - r, mode)
            for r in (1, 7, 15, 16, 17, 33)
            for mode in (
                "shared",
                "different",
                "pairs",
                "quartets",
                "octets",
                "interior",
                "fragmented",
                "empty",
            )
        ]
        cases += [
            (17, n, n - 17, mode)
            for n in (0, 1, 3, 67)
            for mode in ("shared", "empty", "fragmented")
        ]
        for rows, length, base, mode in cases:
            with self.subTest(rows=rows, length=length, mode=mode):
                cpu = self.inputs(rows, length, mode)
                scale = 1 / 16
                ref = self.reference(*cpu, length, base, scale)
                q, k, v, blocks, table = [x.npu() for x in cpu]
                req = torch.tensor([0], dtype=torch.int64, device="npu")
                out, counts, _, _ = qsa_prefill(
                    q,
                    k,
                    v,
                    blocks,
                    table,
                    req,
                    length,
                    base,
                    scale,
                    return_details=True,
                )
                out = out.cpu().float()
                self.assertTrue(torch.isfinite(out).all())
                rel = ((out - ref).norm() / ref.norm().clamp_min(1e-20)).item()
                self.assertLess(rel, 0.01)
                expected = [
                    int(
                        i + 15 < rows
                        and all(
                            torch.equal(cpu[3][i], cpu[3][i + j]) for j in range(1, 16)
                        )
                    )
                    for i in range(0, rows, 16)
                ]
                self.assertEqual(
                    [int(counts[i * 16 + 1].cpu()) for i in range(0, rows, 16)],
                    expected,
                )

    def test_queued_inputs_and_graph_replay(self):
        q, k, v, blocks, table = [x.npu() for x in self.inputs(17, 257, "fragmented")]
        req = torch.tensor([0], dtype=torch.int64, device="npu")
        args = (q, k, v, blocks, table, req, 257, 240)
        ref = qsa_prefill(*args).clone()
        outputs = [
            qsa_prefill(
                q.clone(), k, v, blocks.clone(), table.clone(), req.clone(), 257, 240
            )
            for _ in range(20)
        ]
        torch.npu.synchronize()
        for out in outputs:
            self.assertTrue(torch.equal(out, ref))
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = qsa_prefill(*args)
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.equal(out, ref))

    def test_first_tiling_with_pending_allocator_writes(self):
        # A new cached tiling must be initialized after queued writes to reused
        # device storage, including first use after a different head layout.
        for heads, kv_heads in ((8, 1), (2, 1), (6, 1), (16, 2)):
            for scale in (1 / 19, 1 / 21, 1 / 23):
                with self.subTest(heads=heads, scale=scale):
                    cpu = self.inputs(9, 509, "shared", heads, kv_heads)
                    q, k, v, blocks, table = [x.npu() for x in cpu]
                    req = torch.tensor([0], dtype=torch.int64, device="npu")
                    for size in range(16, 1025, 16):
                        temporary = torch.zeros(size, dtype=torch.uint8, device="npu")
                        del temporary
                    out = qsa_prefill(q, k, v, blocks, table, req, 509, 500, scale)
                    ref = self.reference(*cpu, 509, 500, scale)
                    actual = out.cpu().float()
                    rel = (actual - ref).norm() / ref.norm().clamp_min(1e-20)
                    self.assertLess(rel.item(), 0.01)
                    torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.02)

    def test_interleaved_tiling_shapes_and_scale(self):
        # These row counts previously collided in a hash-only prototype cache.
        # Use empty base selections and nonzero causal tails to keep this
        # regression independent of model weights and large KV fixtures.
        fixtures = []
        for rows, scale in ((3387, 1 / 16), (1691, 1 / 16), (1691, 1 / 8)):
            cpu = self.inputs(rows, 67, "empty")
            ref = self.reference(*cpu, 67, 0, scale)
            q, k, v, blocks, table = [x.npu() for x in cpu]
            req = torch.tensor([0], dtype=torch.int64, device="npu")
            args = (q, k, v, blocks, table, req, 67, 0, scale)
            out, _, base_out, lse = qsa_prefill(*args, return_details=True)
            torch.npu.synchronize()
            self.assertTrue(torch.equal(base_out, torch.zeros_like(base_out)))
            self.assertTrue(torch.isneginf(lse).all())
            rel = (out.cpu().float() - ref).norm() / ref.norm().clamp_min(1e-20)
            self.assertLess(rel.item(), 0.01)
            fixtures.append((args, out.clone()))
        for _ in range(3):
            queued = [(qsa_prefill(*args), ref) for args, ref in fixtures]
            torch.npu.synchronize()
            for out, ref in queued:
                self.assertTrue(torch.equal(out, ref))

    def test_tp_head_layouts(self):
        # Whittle TP2, Whittle TP8 and Qwen3.8-Flash-Next TP4 respectively.
        for heads in (8, 2, 6):
            cases = [
                (rows, 5000, mode)
                for rows in (1, 15, 16, 17, 33)
                for mode in ("shared", "different", "interior", "fragmented", "empty")
            ]
            cases += [
                (17, length, mode)
                for length in (0, 1, 3, 67)
                for mode in ("shared", "fragmented", "empty")
            ]
            cases.append((0, 67, "shared"))
            for rows, length, mode in cases:
                with self.subTest(heads=heads, rows=rows, length=length, mode=mode):
                    cpu = self.inputs(rows, length, mode, heads, 1)
                    ref = self.reference(*cpu, length, length - rows, 1 / 16)
                    q, k, v, blocks, table = [x.npu() for x in cpu]
                    req = torch.tensor([0], dtype=torch.int64, device="npu")
                    out, _, base_out, lse = qsa_prefill(
                        q,
                        k,
                        v,
                        blocks,
                        table,
                        req,
                        length,
                        length - rows,
                        return_details=True,
                    )
                    self.assertEqual(
                        base_out.shape, ((rows + 15) // 16, 16 * heads, 256)
                    )
                    self.assertEqual(lse.shape, ((rows + 15) // 16, 16 * heads))
                    out = out.cpu().float()
                    self.assertTrue(torch.isfinite(out).all())
                    rel = (out - ref).norm() / ref.norm().clamp_min(1e-20)
                    self.assertLess(rel.item(), 0.01)
                    torch.testing.assert_close(out, ref, atol=0.02, rtol=0.02)

    def test_tp_large_prefill(self):
        for heads in (8, 2, 6):
            with self.subTest(heads=heads):
                cpu = self.inputs(1024, 8192, "octets", heads, 1)
                q, k, v, blocks, table = [x.npu() for x in cpu]
                req = torch.tensor([0], dtype=torch.int64, device="npu")
                actual = (
                    qsa_prefill(q, k, v, blocks, table, req, 8192, 7168).cpu().float()
                )
                self.assertTrue(torch.isfinite(actual).all())
                for row in (0, 1, 7, 15, 16, 255, 511, 512, 1023):
                    fixture = (
                        cpu[0][row : row + 1],
                        cpu[1],
                        cpu[2],
                        cpu[3][row : row + 1],
                        cpu[4],
                    )
                    ref = self.reference(*fixture, 8192, 7168 + row, 1 / 16)
                    rel = (actual[row : row + 1] - ref).norm() / ref.norm().clamp_min(
                        1e-20
                    )
                    self.assertLess(rel.item(), 0.01)
                    torch.testing.assert_close(
                        actual[row : row + 1], ref, atol=0.02, rtol=0.02
                    )

    def test_tp_mutating_graph_and_interleaved_heads(self):
        fixtures = []
        for heads, kv_heads in ((8, 1), (2, 1), (6, 1), (16, 2)):
            cpu = self.inputs(17, 257, "fragmented", heads, kv_heads)
            q, k, v, blocks, table = [x.npu() for x in cpu]
            # Exercise a nonzero request row and a non-contiguous table row stride.
            table = torch.cat((table, table, table[:, :1]), 1).repeat(2, 1)[:, :257]
            req = torch.tensor([1], dtype=torch.int64, device="npu")
            args = (q, k, v, blocks, table, req, 257, 240)
            for _ in range(3):
                qsa_prefill(*args)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                out = qsa_prefill(*args)
            fixtures.append((graph, out, args))
        # Reuse each cached tiling after the other head layouts were launched.
        for iteration in range(3):
            for graph, out, args in fixtures:
                q, k, v, blocks, table, req, length, base = args
                cpu = self.inputs(17, length, "fragmented", q.shape[1], k.shape[1])
                if iteration == 1:
                    cpu[3][:] = cpu[3][0].clone()  # shared after unshared capture
                elif iteration == 2:
                    cpu[3].fill_(-1)  # empty base; independent tails still vary
                for device, host in zip((q, k, v, blocks), cpu[:4]):
                    device.copy_(host)
                table[1].copy_(cpu[4][0])
                graph.replay()
                ref = self.reference(*cpu, length, base, 1 / 16)
                actual = out.cpu().float()
                self.assertTrue(torch.isfinite(actual).all())
                rel = (actual - ref).norm() / ref.norm().clamp_min(1e-20)
                self.assertLess(rel.item(), 0.01)
                torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.02)
                self.assertTrue(torch.equal(out, qsa_prefill(*args)))

    def test_tp_shards_reconstruct_global_attention(self):
        # Evaluate every rank sequentially on one NPU. This verifies head/KV
        # replication semantics, not a multi-device communication deployment.
        for global_heads, tp in ((16, 2), (16, 8), (24, 4)):
            with self.subTest(global_heads=global_heads, tp=tp):
                cpu = self.inputs(17, 257, "fragmented", global_heads, 2)
                q, k, v, blocks, table = cpu
                ref = self.reference(*cpu, 257, 240, 1 / 16)
                req = torch.tensor([0], dtype=torch.int64, device="npu")
                device_blocks, device_table = blocks.npu(), table.npu()
                shards = []
                local_heads = global_heads // tp
                for rank in range(tp):
                    kv_head = rank // (tp // 2)
                    shards.append(
                        qsa_prefill(
                            q[:, rank * local_heads : (rank + 1) * local_heads]
                            .contiguous()
                            .npu(),
                            k[:, kv_head : kv_head + 1].contiguous().npu(),
                            v[:, kv_head : kv_head + 1].contiguous().npu(),
                            device_blocks,
                            device_table,
                            req,
                            257,
                            240,
                        )
                        .cpu()
                        .float()
                    )
                actual = torch.cat(shards, 1)
                rel = (actual - ref).norm() / ref.norm().clamp_min(1e-20)
                self.assertLess(rel.item(), 0.01)
                torch.testing.assert_close(actual, ref, atol=0.02, rtol=0.02)

    def test_unsupported_head_layouts(self):
        req = torch.tensor([0], dtype=torch.int64, device="npu")
        for heads, kv_heads in ((4, 1), (6, 2), (16, 1), (8, 2), (0, 1)):
            with self.subTest(heads=heads, kv_heads=kv_heads):
                q, k, v, blocks, table = [
                    x.npu() for x in self.inputs(1, 67, "shared", heads, kv_heads)
                ]
                with self.assertRaisesRegex(ValueError, "unsupported Q/KV heads"):
                    qsa_prefill(q, k, v, blocks, table, req, 67, 66)
                packed = torch.empty(
                    (1, 16 * heads, 256), dtype=q.dtype, device=q.device
                )
                slots = torch.zeros((1, 2064), dtype=torch.int32, device=q.device)
                counts = torch.zeros((16,), dtype=torch.int32, device=q.device)
                with self.assertRaisesRegex(
                    RuntimeError, "unsupported packed Q/KV heads"
                ):
                    torch.ops.npu.qsa_prefill_runs(
                        packed, k, v, slots, slots, counts, 1 / 16
                    )

    def test_zero_rows_and_invalid_arguments(self):
        q, k, v, blocks, table = [x.npu() for x in self.inputs(0, 67, "shared")]
        req = torch.tensor([0], dtype=torch.int64, device="npu")
        self.assertEqual(
            qsa_prefill(q, k, v, blocks, table, req, 67, 67).shape, q.shape
        )
        q, k, v, blocks, table = [x.npu() for x in self.inputs(1, 67, "shared")]
        args = (q, k, v, blocks, table, req, 67, 66)
        for scale in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                qsa_prefill(*args, scale=scale)
        with self.assertRaises(ValueError):
            qsa_prefill(q.float(), *args[1:])
        with self.assertRaises(ValueError):
            qsa_prefill(q, k, v, blocks[:, :511], table, req, 67, 66)
        with self.assertRaises(ValueError):
            qsa_prefill(q, k, v, blocks, table, req, 68, 66)


if __name__ == "__main__":
    unittest.main()
