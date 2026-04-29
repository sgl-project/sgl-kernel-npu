"""apply_token_bitmask 功能与性能测试

通过 sgl-kernel-npu 接口 (torch.ops.npu.apply_token_bitmask) 测试算子。

用法:
    # 功能测试 (全部)
    python test_apply_token_bitmask_sgl.py

    # 性能测试
    python test_apply_token_bitmask_sgl.py --perf

    # 仅运行特定类别
    python test_apply_token_bitmask_sgl.py --category boundary
    python test_apply_token_bitmask_sgl.py --category llm
    python test_apply_token_bitmask_sgl.py --category general
"""

import argparse
import math
import sys
import time
import unittest

import sgl_kernel_npu
import torch
import torch_npu

# ---------------------------------------------------------------------------
# CPU reference implementation
# ---------------------------------------------------------------------------


def apply_token_bitmask_ref(logits, bitmask, indices=None):
    """CPU reference: for each bit=0 in bitmask, set corresponding logit to -inf."""
    if indices is not None:
        rows = indices.tolist()
    else:
        rows = range(logits.shape[0])

    vocab_size = logits.shape[1]
    for row in rows:
        bm = bitmask[row]
        for bm_i in range(bm.shape[0]):
            bm_val = bm[bm_i].item()
            start_col = bm_i * 32
            end_col = min(start_col + 32, vocab_size)
            for bit in range(end_col - start_col):
                if not ((bm_val >> bit) & 1):
                    logits[row, start_col + bit] = float("-inf")
    return logits


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def make_bitmask(batch, vocab_size, mode="random"):
    """Create bitmask tensor.  mode: random | all_masked | all_unmasked | half_masked"""
    bitmask_w = math.ceil(vocab_size / 32)
    if mode == "all_masked":
        bitmask = torch.zeros(batch, bitmask_w, dtype=torch.int32)
    elif mode == "all_unmasked":
        # -1 in two's complement int32 = 0xFFFFFFFF (all bits set)
        bitmask = torch.full((batch, bitmask_w), -1, dtype=torch.int32)
    elif mode == "half_masked":
        # 0x55555555 = 1431655765, 0xAAAAAAAA = -1431655766 in signed int32
        val_a = torch.tensor([1431655765], dtype=torch.int32)
        val_b = torch.tensor([-1431655766], dtype=torch.int32)
        bitmask = torch.zeros(batch, bitmask_w, dtype=torch.int32)
        for i in range(bitmask_w):
            bitmask[:, i] = val_a if i % 2 == 0 else val_b
    else:
        # Generate random 32-bit bitmask via int64 then cast to int32
        bitmask = torch.randint(0, 2**31, (batch, bitmask_w), dtype=torch.int64)
        bitmask = bitmask.to(torch.int32)
        # Set top bit randomly too
        top_bit = torch.randint(0, 2, (batch, bitmask_w), dtype=torch.int32)
        bitmask = bitmask.bitwise_or(top_bit << 31)
    # Mask out bits beyond vocab_size in the last group
    remainder = vocab_size % 32
    if remainder > 0:
        bitmask[:, -1] &= (1 << remainder) - 1
    return bitmask


# ---------------------------------------------------------------------------
# Helper: run NPU and compare with CPU reference
# ---------------------------------------------------------------------------


def _run_and_compare(logits, bitmask, indices=None, label=""):
    """Run NPU op, compare with CPU ref, return True if pass."""
    vocab_size = logits.shape[1]

    # CPU reference
    logits_ref = logits.clone().float()
    apply_token_bitmask_ref(logits_ref, bitmask, indices)

    # NPU
    logits_npu = logits.npu().contiguous()
    bitmask_npu = bitmask.npu().contiguous()
    if indices is not None:
        indices_npu = indices.npu()
        out_npu = torch.ops.npu.apply_token_bitmask(
            logits_npu, bitmask_npu, indices_npu
        )
    else:
        out_npu = torch.ops.npu.apply_token_bitmask(logits_npu, bitmask_npu)
    out_cpu = out_npu.float().cpu()

    # Compare masked positions
    ref_masked = logits_ref == float("-inf")
    res_masked = out_cpu == float("-inf")
    if not torch.equal(ref_masked, res_masked):
        print(f"  FAIL [{label}]: masked position mismatch")
        return False

    # Compare non-masked values
    ref_vals = logits_ref.clone()
    res_vals = out_cpu.clone()
    if ref_masked.any():
        ref_vals[ref_masked] = 0.0
        res_vals[res_masked] = 0.0

    try:
        torch.testing.assert_close(res_vals, ref_vals, atol=1e-2, rtol=1e-2)
        return True
    except AssertionError as e:
        print(f"  FAIL [{label}]: value mismatch - {e}")
        return False


# ---------------------------------------------------------------------------
# Functionality tests
# ---------------------------------------------------------------------------

SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

LLM_SHAPES = [
    ("small", (1, 320)),
    ("small", (4, 320)),
    ("tiny", (1, 64)),
    ("base", (1, 32000)),
    ("base", (8, 32000)),
    ("large", (2, 128256)),
    ("common", (16, 32000)),
]

GENERAL_SHAPES = [
    ("sub1", (2, 1)),
    ("sub32", (2, 31)),
    ("eq32", (2, 32)),
    ("over32", (2, 33)),
    ("unaligned", (4, 100)),
    ("stress", (32, 32000)),
    ("ularge", (8, 32100)),
]


class TestApplyTokenBitmaskFunction(unittest.TestCase):
    """Functional correctness tests."""

    @torch.no_grad()
    def test_llm_shapes(self):
        """LLM inference typical shapes x 3 dtypes = 21 cases."""
        passed, total = 0, 0
        for dtype in SUPPORTED_DTYPES:
            for tag, (batch, vocab) in LLM_SHAPES:
                total += 1
                logits = torch.randn(batch, vocab, dtype=dtype)
                bitmask = make_bitmask(batch, vocab, "random")
                label = f"llm/{tag}/{batch}x{vocab}/{dtype}"
                if _run_and_compare(logits, bitmask, label=label):
                    passed += 1
        print(f"\n  LLM shapes: {passed}/{total} passed")
        self.assertEqual(passed, total)

    @torch.no_grad()
    def test_general_shapes(self):
        """Boundary and non-aligned shapes x 3 dtypes = 21 cases."""
        passed, total = 0, 0
        for dtype in SUPPORTED_DTYPES:
            for tag, (batch, vocab) in GENERAL_SHAPES:
                total += 1
                logits = torch.randn(batch, vocab, dtype=dtype)
                bitmask = make_bitmask(batch, vocab, "random")
                label = f"general/{tag}/{batch}x{vocab}/{dtype}"
                if _run_and_compare(logits, bitmask, label=label):
                    passed += 1
        print(f"\n  General shapes: {passed}/{total} passed")
        self.assertEqual(passed, total)

    @torch.no_grad()
    def test_all_masked(self):
        """All bits = 0: every logit becomes -inf."""
        for dtype in SUPPORTED_DTYPES:
            logits = torch.randn(4, 320, dtype=dtype)
            bitmask = make_bitmask(4, 320, "all_masked")
            logits_npu = logits.npu().contiguous()
            bitmask_npu = bitmask.npu().contiguous()
            out = (
                torch.ops.npu.apply_token_bitmask(logits_npu, bitmask_npu).float().cpu()
            )
            self.assertTrue(
                torch.all(out == float("-inf")), f"all_masked failed for {dtype}"
            )

    @torch.no_grad()
    def test_all_unmasked(self):
        """All bits = 1: logits unchanged."""
        for dtype in SUPPORTED_DTYPES:
            logits = torch.randn(4, 320, dtype=dtype)
            bitmask = make_bitmask(4, 320, "all_unmasked")
            logits_npu = logits.npu().contiguous()
            bitmask_npu = bitmask.npu().contiguous()
            out = (
                torch.ops.npu.apply_token_bitmask(logits_npu, bitmask_npu).float().cpu()
            )
            torch.testing.assert_close(out, logits.float(), atol=1e-2, rtol=1e-2)

    @torch.no_grad()
    def test_half_masked(self):
        """Alternating 0x55555555 / 0xAAAAAAAA pattern."""
        for dtype in SUPPORTED_DTYPES:
            logits = torch.randn(4, 320, dtype=dtype)
            bitmask = make_bitmask(4, 320, "half_masked")
            self.assertTrue(
                _run_and_compare(logits, bitmask, label=f"half_masked/{dtype}")
            )

    @torch.no_grad()
    def test_single_row(self):
        """batch=1 edge case."""
        for dtype in SUPPORTED_DTYPES:
            logits = torch.randn(1, 128, dtype=dtype)
            bitmask = make_bitmask(1, 128, "random")
            self.assertTrue(
                _run_and_compare(logits, bitmask, label=f"single_row/{dtype}")
            )

    @torch.no_grad()
    def test_with_indices(self):
        """Optional indices: only specified rows are modified."""
        for dtype in SUPPORTED_DTYPES:
            batch, vocab = 8, 320
            logits = torch.randn(batch, vocab, dtype=dtype)
            bitmask = make_bitmask(batch, vocab, "random")
            indices = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
            self.assertTrue(
                _run_and_compare(
                    logits, bitmask, indices=indices, label=f"indices/{dtype}"
                )
            )

    @torch.no_grad()
    def test_indices_all_rows(self):
        """indices covering all rows == no indices."""
        for dtype in SUPPORTED_DTYPES:
            batch, vocab = 4, 320
            logits = torch.randn(batch, vocab, dtype=dtype)
            bitmask = make_bitmask(batch, vocab, "random")
            indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

            logits_npu1 = logits.npu().contiguous()
            bitmask_npu = bitmask.npu().contiguous()
            out1 = (
                torch.ops.npu.apply_token_bitmask(logits_npu1, bitmask_npu)
                .float()
                .cpu()
            )

            logits_npu2 = logits.npu().contiguous()
            out2 = (
                torch.ops.npu.apply_token_bitmask(
                    logits_npu2, bitmask_npu, indices.npu()
                )
                .float()
                .cpu()
            )

            torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


def run_perf_test(batch, vocab, dtype, warmup=5, iters=50, indices=None):
    """Benchmark apply_token_bitmask, return avg latency in ms."""
    bitmask = make_bitmask(batch, vocab, "random")

    logits_npu = torch.randn(batch, vocab, dtype=dtype, device="npu").contiguous()
    bitmask_npu = bitmask.to("npu").contiguous()
    indices_npu = indices.to("npu") if indices is not None else None

    # Warmup
    for _ in range(warmup):
        if indices_npu is not None:
            torch.ops.npu.apply_token_bitmask(logits_npu, bitmask_npu, indices_npu)
        else:
            torch.ops.npu.apply_token_bitmask(logits_npu, bitmask_npu)
    torch.npu.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        if indices_npu is not None:
            torch.ops.npu.apply_token_bitmask(logits_npu, bitmask_npu, indices_npu)
        else:
            torch.ops.npu.apply_token_bitmask(logits_npu, bitmask_npu)
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    return elapsed


def perf_suite():
    """Run performance benchmarks and print results."""
    print("=" * 80)
    print("Performance Benchmark: apply_token_bitmask")
    print("=" * 80)

    configs = [
        # (label, batch, vocab, dtype)
        ("decode-typical", 16, 32000, torch.float16),
        ("decode-large", 64, 32000, torch.float16),
        ("prefill-base", 1, 128256, torch.float16),
        ("prefill-large", 4, 128256, torch.float16),
        ("decode-bf16", 16, 32000, torch.bfloat16),
        ("decode-fp32", 16, 32000, torch.float32),
        ("small", 1, 320, torch.float16),
    ]

    print(
        f"\n{'Config':<20s} {'Batch':>5s} {'Vocab':>7s} {'Dtype':>8s} "
        f"{'Latency(ms)':>12s} {'Bandwidth(GB/s)':>16s}"
    )
    print("-" * 80)

    for label, batch, vocab, dtype in configs:
        try:
            latency = run_perf_test(batch, vocab, dtype)
            total_bytes = batch * vocab * 4 * 2  # approx: read logits + mask
            bw = total_bytes / (latency * 1e-3) / 1e9 if latency > 0 else 0
            dtype_str = str(dtype).replace("torch.", "")
            print(
                f"{label:<20s} {batch:>5d} {vocab:>7d} {dtype_str:>8s} "
                f"{latency:>12.3f} {bw:>16.2f}"
            )
        except Exception as e:
            print(f"{label:<20s} {batch:>5d} {vocab:>7d} {'ERROR':>8s} {str(e):>12s}")

    # With indices
    print(f"\n--- With Indices ---")
    for batch, vocab in [(16, 32000), (64, 32000)]:
        num_idx = max(1, batch // 2)
        indices = torch.randperm(batch, dtype=torch.int32)[:num_idx]
        try:
            latency = run_perf_test(batch, vocab, torch.float16, indices=indices)
            print(
                f"  indices={num_idx:>3d}  batch={batch:>3d}  vocab={vocab:>6d}  "
                f"latency={latency:.3f} ms"
            )
        except Exception as e:
            print(
                f"  indices={num_idx:>3d}  batch={batch:>3d}  vocab={vocab:>6d}  ERROR: {e}"
            )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="apply_token_bitmask test")
    parser.add_argument(
        "--perf", action="store_true", help="Run performance benchmark only"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Run specific category: boundary | llm | general | all",
    )
    args = parser.parse_args()

    if args.perf:
        perf_suite()
        sys.exit(0)

    # Functional tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if args.category == "boundary":
        for name in [
            "test_all_masked",
            "test_all_unmasked",
            "test_half_masked",
            "test_single_row",
            "test_with_indices",
            "test_indices_all_rows",
        ]:
            suite.addTests(
                loader.loadTestsFromName(
                    f"test_apply_token_bitmask_sgl.TestApplyTokenBitmaskFunction.{name}"
                )
            )
    elif args.category == "llm":
        suite.addTests(
            loader.loadTestsFromName(
                "test_apply_token_bitmask_sgl.TestApplyTokenBitmaskFunction.test_llm_shapes"
            )
        )
    elif args.category == "general":
        suite.addTests(
            loader.loadTestsFromName(
                "test_apply_token_bitmask_sgl.TestApplyTokenBitmaskFunction.test_general_shapes"
            )
        )
    else:
        suite.addTests(loader.loadTestsFromTestCase(TestApplyTokenBitmaskFunction))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print(f"\n{'='*60}")
        print(f"ALL {result.testsRun} TESTS PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{len(result.failures)} failures, {len(result.errors)} errors")
