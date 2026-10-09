"""Functional tests for the ELU AscendC operator sample.

The operator is exposed through the sgl-kernel-npu extension as
``torch.ops.npu.elu(x, alpha)`` (see ``csrc/elu/`` and the ELU sample README).

Run:
    python tests/python/sgl_kernel_npu/test_elu.py
"""

import unittest

import sgl_kernel_npu  # noqa: F401  loads libsgl_kernel_npu.so
import torch
import torch_npu  # noqa: F401  makes torch.ops.npu available (PrivateUse1)

# ---------------------------------------------------------------------------
# CPU reference implementation
# ---------------------------------------------------------------------------


def elu_ref(x, alpha):
    """CPU reference: elu(x) = x if x > 0 else alpha * (exp(x) - 1)."""
    return torch.where(x > 0, x, alpha * torch.expm1(x))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestElu(unittest.TestCase):
    def _run_case(self, shape, dtype, alpha):
        torch.manual_seed(0)
        x = (torch.randn(*shape, dtype=torch.float32) * 2.0).to(dtype)
        y = torch.ops.npu.elu(x.npu(), alpha).cpu()
        # The output must preserve the input's shape (and stay contiguous).
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        ref = elu_ref(x, alpha)
        if dtype == torch.float16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-5, 1e-5
        torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)

    def test_fp32_basic(self):
        # Large aligned input (kernel fast path, no host padding).
        self._run_case((4096,), torch.float32, 1.0)

    def test_fp32_arbitrary_length(self):
        # Odd lengths exercise the host-side alignment/padding path.
        # Multi-dimensional non-aligned inputs also hit the padding path, which
        # must flatten before copying and restore the original shape on return.
        self._run_case((777,), torch.float32, 1.0)
        self._run_case((5, 7, 2), torch.float32, 1.0)
        self._run_case((3, 7, 5), torch.float32, 1.0)

    def test_fp32_tiny_input(self):
        # Tensors smaller than one tile are padded to a full tile.
        self._run_case((1,), torch.float32, 1.0)

    def test_fp32_custom_alpha(self):
        self._run_case((1024, 4), torch.float32, 1.67326)

    def test_fp32_wide_range(self):
        # Large positive and very negative values (exp underflows to 0).
        x = torch.linspace(-30.0, 30.0, 2048, dtype=torch.float32)
        y = torch.ops.npu.elu(x.npu(), 1.0).cpu()
        ref = elu_ref(x, 1.0)
        torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)

    def test_fp16_basic(self):
        self._run_case((4096,), torch.float16, 1.0)

    def test_fp16_arbitrary_length(self):
        self._run_case((32001,), torch.float16, 1.0)

    def test_fp16_custom_alpha(self):
        self._run_case((2, 128, 64), torch.float16, 0.5)

    def test_empty_input(self):
        x = torch.empty(0, dtype=torch.float32, device="npu")
        y = torch.ops.npu.elu(x, 1.0)
        self.assertEqual(y.numel(), 0)

    def test_unsupported_dtype_raises(self):
        x = torch.ones(8, dtype=torch.int32, device="npu")
        with self.assertRaises(RuntimeError):
            torch.ops.npu.elu(x, 1.0)


if __name__ == "__main__":
    unittest.main()
