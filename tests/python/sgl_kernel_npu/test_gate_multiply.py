import unittest

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.fla.gate_multiply import gate_multiply


@unittest.skipUnless(torch.npu.is_available(), "NPU required")
class TestGateMultiply(unittest.TestCase):
    def test_exact_and_queued(self):
        torch.manual_seed(100326)
        for n in (0, 1, 4097, 16777215, 16777216, 16777217, 16384 * 4096):
            with self.subTest(n=n):
                y = torch.randn(n, device="npu", dtype=torch.bfloat16)
                gate = torch.sigmoid(torch.randn(n, device="npu") * 20)
                ref = (y.float() * gate).to(y.dtype)
                queued = [gate_multiply(y.clone(), gate.clone()) for _ in range(3)]
                torch.npu.synchronize()
                for out in queued:
                    self.assertTrue(torch.equal(ref, out))

    def test_strided_and_special_values(self):
        y = torch.randn(8192, 4098, device="npu", dtype=torch.bfloat16)[:, ::2]
        gate = torch.sigmoid(torch.randn(8192, 4098, device="npu"))[:, ::2]
        self.assertTrue(
            torch.equal(gate_multiply(y, gate), (y.float() * gate).to(y.dtype))
        )
        y = torch.ones(16777216, device="npu", dtype=torch.bfloat16)
        gate = torch.ones_like(y, dtype=torch.float32)
        y[:8] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0, 0.5],
            device="npu",
            dtype=y.dtype,
        )
        gate[:8] = torch.tensor(
            [1.0, 1.0, 0.0, -1.0, 1.0, float("nan"), float("inf"), 0.0], device="npu"
        )
        ref = (y.float() * gate).to(y.dtype)
        out = gate_multiply(y, gate)
        self.assertTrue(torch.equal(torch.isnan(out), torch.isnan(ref)))
        mask = ~torch.isnan(ref)
        self.assertTrue(
            torch.equal(out[mask].view(torch.int16), ref[mask].view(torch.int16))
        )

    def test_graph(self):
        y = torch.randn(16777216, device="npu", dtype=torch.bfloat16)
        gate = torch.sigmoid(torch.randn_like(y, dtype=torch.float32))
        for _ in range(3):
            gate_multiply(y, gate)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = gate_multiply(y, gate)
        for _ in range(5):
            y.normal_()
            gate.uniform_()
            graph.replay()
            torch.npu.synchronize()
            self.assertTrue(torch.equal(out, (y.float() * gate).to(y.dtype)))

    def test_guards(self):
        y = torch.ones(8, device="npu", dtype=torch.bfloat16)
        gate = torch.ones(8, device="npu")
        for a, b in (
            (y.float(), gate),
            (y, gate.bfloat16()),
            (y, gate[:4]),
            (y.cpu(), gate.cpu()),
            (y.requires_grad_(), gate),
        ):
            with self.assertRaises(ValueError):
                gate_multiply(a, b)


if __name__ == "__main__":
    unittest.main()
