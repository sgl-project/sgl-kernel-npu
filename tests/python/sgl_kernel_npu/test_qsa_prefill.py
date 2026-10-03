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

    def inputs(self, rows, length, mode):
        capacity = max(6000, length)
        q = torch.randn(rows, 16, 256, dtype=torch.bfloat16)
        k = torch.randn(capacity, 2, 256, dtype=torch.bfloat16)
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
                keys = k[slots].float().repeat_interleave(8, 1)
                values = v[slots].float().repeat_interleave(8, 1)
                logits = torch.einsum("hd,shd->hs", q[row].float(), keys) * scale
                outputs.append(torch.einsum("hs,shd->hd", logits.softmax(-1), values))
            else:
                outputs.append(torch.zeros(16, 256))
        return torch.stack(outputs) if outputs else torch.empty(0, 16, 256)

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
