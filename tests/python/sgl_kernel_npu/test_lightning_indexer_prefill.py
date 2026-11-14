import unittest

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.indexer.lightning_indexer_prefill import lightning_indexer_prefill


def independent_reference(q, k, n):
    rows, keys = len(q), len(k)
    if not rows or not keys:
        return torch.full((rows, 512), -1, dtype=torch.int32, device=q.device)
    counts = (
        torch.arange(rows, device=q.device, dtype=torch.int32) + n - rows + 1
    ) // 4
    width = min(keys, 512)
    cache = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, (-keys) % 64)).reshape(
        -1, 64, 1, 128
    )
    table = (
        torch.arange(len(cache), device=q.device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(rows, -1)
        .contiguous()
    )
    result = torch_npu.npu_lightning_indexer(
        q.unsqueeze(1),
        cache,
        torch.ones((rows, 1, 4), device=q.device, dtype=q.dtype),
        actual_seq_lengths_query=torch.ones(rows, device=q.device, dtype=torch.int32),
        actual_seq_lengths_key=counts.clamp_min(1),
        block_table=table,
        layout_query="BSND",
        layout_key="PA_BSND",
        sparse_count=width,
        sparse_mode=0,
    )
    out = (result[0] if isinstance(result, tuple) else result).reshape(rows, width)
    out = torch.where(
        torch.arange(width, device=q.device).unsqueeze(0) < counts.unsqueeze(1), out, -1
    )
    return torch.nn.functional.pad(out, (0, 512 - width), value=-1).to(torch.int32)


@unittest.skipUnless(torch.npu.is_available(), "NPU required")
class TestLightningIndexerPrefill(unittest.TestCase):
    def test_ordered_indices(self):
        torch.manual_seed(4201)
        cases = [(0, 0), (1, 1), (3, 3), (7, 7), (65, 2052), (1023, 4096), (1024, 2048)]
        cases += [(1024 + phase, 4096) for phase in range(4)]
        cases += [(1024, 4096 + phase) for phase in range(4)]
        cases += [(16384, 16384), (16384, 32768), (8313, 44408)]
        for rows, n in cases:
            for zero in (False, True):
                with self.subTest(rows=rows, length=n, zero=zero):
                    q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
                    k = torch.randn(n // 4, 1, 128, device="npu", dtype=torch.bfloat16)
                    if zero:
                        q.zero_()
                        k.zero_()
                    ref = independent_reference(q, k, n)
                    queued = [
                        lightning_indexer_prefill(
                            q.clone(), k.clone(), sequence_length=n
                        )
                        for _ in range(3)
                    ]
                    torch.npu.synchronize()
                    for out in queued:
                        self.assertTrue(torch.equal(out, ref))
                    counts = (torch.arange(rows, device=q.device) + n - rows + 1) // 4
                    self.assertTrue(
                        torch.equal((ref >= 0).sum(1), counts.clamp(max=512))
                    )
                    self.assertTrue(bool(((ref < n // 4) | (ref == -1)).all()))

    def test_metadata_guards(self):
        q = torch.zeros(1024, 4, 128, device="npu", dtype=torch.bfloat16)
        k = torch.zeros(512, 1, 128, device="npu", dtype=torch.bfloat16)
        for n in (-1, 0, 2044, 2048.0, True, 2**31):
            with self.assertRaises(ValueError):
                lightning_indexer_prefill(q, k, sequence_length=n)
        with self.assertRaises(ValueError):
            lightning_indexer_prefill(q.float(), k, sequence_length=2048)
        with self.assertRaises(ValueError):
            lightning_indexer_prefill(q[:, :, ::2], k, sequence_length=2048)
        with self.assertRaises(ValueError):
            lightning_indexer_prefill(q.cpu(), k.cpu(), sequence_length=2048)


if __name__ == "__main__":
    unittest.main()
