import random
import time
import unittest

import numpy as np
import sgl_kernel_npu
import torch
import torch_npu

torch.set_printoptions(threshold=float("inf"))


class TestMatrixMultiplication(unittest.TestCase):

    def compute_golden(self, a, b, res1):
        """Compute reference result (golden)"""
        torch.matmul(a, b, out=res1)

    def assert_tensors_almost_equal(self, actual, expected, dtype):
        """Check if two tensors are approximately equal (considering floating point errors)"""
        self.assertEqual(actual.shape, expected.shape, "Shape mismatch")

        # Check for NaN
        self.assertFalse(torch.isnan(actual).any(), "Actual result contains NaN")
        self.assertFalse(torch.isnan(expected).any(), "Expected result contains NaN")

        # Check for Inf
        self.assertFalse(torch.isinf(actual).any(), "Actual result contains Inf")
        self.assertFalse(torch.isinf(expected).any(), "Expected result contains Inf")

        # Set different tolerances based on data type
        if dtype == torch.float16:
            rtol, atol = 5e-4, 5e-4
        else:  # bfloat16
            rtol, atol = 1e-3, 1e-3

        # Compare values
        diff = torch.abs(actual - expected)
        max_diff = diff.max().item()
        max_expected = torch.abs(expected).max().item()

        # Check relative and absolute errors
        if max_expected > 0:
            relative_diff = max_diff / max_expected
            self.assertLessEqual(
                relative_diff,
                rtol,
                f"Relative error too large: {relative_diff} > {rtol}. Max difference: {max_diff}",
            )

        self.assertLessEqual(
            max_diff, atol, f"Absolute error too large: {max_diff} > {atol}"
        )

    def test_boundary_conditions(self):
        """Test boundary conditions"""
        test_cases = [
            # (m, k, n)
            (1, 1, 1),  # Minimum size
            (10, 1, 1),  # b=1
            (1, 1, 10),  # m=1
            (5, 1, 5),  # k=1
            (2, 2, 1),  # n=1
            (1, 1, 100),  # Flat case
            (100, 100, 1),  # Flat case
            (3, 4, 5),  # Random small size
            (20, 30, 40),  # Medium size
            (128, 512, 128),  # target case
            (160, 512, 128),
        ]

        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            for m, k, n in test_cases:
                with self.subTest(dtype=dtype, shape=f"({m}, {k}, {n})"):
                    a = torch.randn(m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((m, n), dtype=dtype, device="npu")
                    res2 = torch.empty((m, n), dtype=dtype, device="npu")

                    self.compute_golden(a, b_tensor, res1)
                    torch.ops.npu.catlass_matmul_basic(a, b_tensor, res2)
                    self.assert_tensors_almost_equal(res1, res2, dtype)

    def test_random_shapes(self):
        """Test randomly generated shapes"""
        num_tests = 1
        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            for _ in range(num_tests):
                # Generate reasonable random sizes
                m = random.randint(1, 500)
                k = random.randint(1, 500)
                n = random.randint(1, 500)

                with self.subTest(dtype=dtype, shape=f"Random ({m}, {k}, {n})"):
                    a = torch.randn(m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((m, n), dtype=dtype, device="npu")
                    res2 = torch.empty((m, n), dtype=dtype, device="npu")

                    self.compute_golden(a, b_tensor, res1)
                    torch.ops.npu.catlass_matmul_basic(a, b_tensor, res2)
                    self.assert_tensors_almost_equal(res1, res2, dtype)

    def test_zero_values(self):
        """Test zero input values"""
        dtypes = [torch.float16, torch.bfloat16]
        m, k, n = 4, 3, 2

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a = torch.zeros(m, k, dtype=dtype, device="npu")
                b_tensor = torch.zeros(k, n, dtype=dtype, device="npu")
                res1 = torch.empty((m, n), dtype=dtype, device="npu")
                res2 = torch.empty((m, n), dtype=dtype, device="npu")

                self.compute_golden(a, b_tensor, res1)
                torch.ops.npu.catlass_matmul_basic(a, b_tensor, res2)
                self.assert_tensors_almost_equal(res1, res2, dtype)
                self.assertTrue(torch.all(res2 == 0))


if __name__ == "__main__":
    try:
        catlass_ops = torch.ops.npu.catlass_matmul_basic
    except Exception as e:
        print("use catlass ops in sglang-kernel need to set BUILD_KERNELS_MODULE in cmake during compiling")
        raise e

    unittest.main(verbosity=2)
