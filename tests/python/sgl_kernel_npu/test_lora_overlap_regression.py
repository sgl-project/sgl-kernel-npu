"""Numerical coverage for LoRA buffers reused by overlap loading."""

import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401


class TestLoRAOverlapKernels(unittest.TestCase):
    def test_padded_ranks_and_fused_slices(self):
        torch.set_num_threads(1)
        for dtype in (torch.float16, torch.bfloat16):
            for widths in ([1024], [4096], [2048, 1024, 1024], [3072, 3072]):
                with self.subTest(dtype=dtype, widths=widths):
                    self._check_projection(dtype, widths)

    def _check_projection(self, dtype, widths):
        generator = torch.Generator().manual_seed(42)
        ranks = [8, 16, 32, 64, 0]
        scales = [0.25, 1.0, 2.0, 0.5, 0.0]
        indices = [0, 1, -1, 2, 3, 4]
        lengths = [1, 3, 2, 1, 2, 1]
        max_rank, hidden = max(ranks), 1024
        slices, tokens = len(widths), sum(lengths)
        x = torch.randn(tokens, hidden, generator=generator).to(dtype)
        a = (
            torch.randn(len(ranks), max_rank * slices, hidden, generator=generator)
            * 0.02
        ).to(dtype)
        b = (
            torch.randn(len(ranks), sum(widths), max_rank, generator=generator) * 0.02
        ).to(dtype)
        base = (torch.randn(tokens, sum(widths), generator=generator) * 0.1).to(dtype)
        # Unused columns model dirty padding left after a larger adapter was
        # evicted. Their values must never affect a smaller adapter's result.
        for slot, rank in enumerate(ranks):
            b[slot, :, rank:] = 17
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1] + width)

        expected_mid = torch.zeros(tokens, max_rank * slices)
        expected = base.float().clone()
        start = 0
        for slot, length in zip(indices, lengths):
            end = start + length
            if slot >= 0 and ranks[slot] > 0:
                rank = ranks[slot]
                mid = (
                    x[start:end].float() @ a[slot, : rank * slices].float().T
                ) * scales[slot]
                expected_mid[start:end, : rank * slices] = mid
                for part, (left, right) in enumerate(zip(offsets, offsets[1:])):
                    expected[start:end, left:right] += (
                        mid[:, part * rank : (part + 1) * rank]
                        @ b[slot, left:right, :rank].float().T
                    )
            start = end

        def int_tensor(values):
            return torch.tensor(values, dtype=torch.int32, device="npu")

        npu_indices, npu_lengths = int_tensor(indices), int_tensor(lengths)
        npu_ranks = int_tensor(ranks)
        actual_mid = torch.zeros(tokens, max_rank * slices, device="npu")
        actual = base.npu()
        torch.ops.npu.sgemmv_shrink(
            x.npu(),
            a.npu(),
            npu_indices,
            npu_lengths,
            npu_ranks * slices,
            torch.tensor(scales, dtype=torch.float16, device="npu"),
            actual_mid,
        )
        torch.ops.npu.sgemmv_expand(
            actual_mid,
            b.npu(),
            npu_indices,
            npu_lengths,
            npu_ranks,
            int_tensor(offsets),
            actual,
        )
        # Only the packed active rows are defined by shrink; expand must not
        # consume its unused tail either.
        start = 0
        for slot, length in zip(indices, lengths):
            if slot >= 0 and ranks[slot] > 0:
                stop = ranks[slot] * slices
                torch.testing.assert_close(
                    actual_mid[start : start + length, :stop].cpu(),
                    expected_mid[start : start + length, :stop],
                    atol=1e-5,
                    rtol=1e-4,
                )
            start += length
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual.cpu(),
            expected.to(dtype),
            atol=1e-3,
            rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
        )
        print(
            f"{dtype}, slices={slices}, widths={widths}: max error="
            f"{(actual.cpu().float() - expected.to(dtype).float()).abs().max().item():.8f}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
