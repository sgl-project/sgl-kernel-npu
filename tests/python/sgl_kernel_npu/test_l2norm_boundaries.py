import unittest

import torch
import torch_npu
from sgl_kernel_npu.fla.l2norm import l2norm
from sgl_kernel_npu.utils.triton_utils import get_device_properties


class TestL2NormBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch_npu.npu.is_available():
            raise unittest.SkipTest("an Ascend NPU is required")
        torch_npu.npu.set_device(0)

    def setUp(self):
        torch.manual_seed(123)

    def check_output(self, x, eps=1e-6, output_dtype=None):
        actual = l2norm(x, eps=eps, output_dtype=output_dtype)
        ref = x.cpu().float()
        expected = ref * torch.rsqrt((ref * ref).sum(dim=-1, keepdim=True) + eps)
        dtype = x.dtype if output_dtype is None else output_dtype
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, dtype)
        tolerances = {
            torch.float16: (1e-3, 1e-4),
            torch.bfloat16: (8e-3, 1e-3),
            torch.float32: (1e-5, 1e-6),
        }
        rtol, atol = tolerances[dtype]
        torch.testing.assert_close(
            actual.cpu(), expected.to(dtype), rtol=rtol, atol=atol
        )

    def test_dimension_and_row_boundaries(self):
        num_cores = get_device_properties()[1]
        # Cross both padded-width transitions and the fallback boundary.
        # The large row count exercises multiple tiles per core plus a tail.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for dim in (128, 129, 256, 257, 511, 512, 513):
                for rows in (1, 17, num_cores * 109 + 1):
                    with self.subTest(dtype=dtype, dim=dim, rows=rows):
                        x = torch.randn(rows, dim, device="npu", dtype=dtype)
                        self.check_output(x)

    def test_output_dtype_and_shape(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for output_dtype in (torch.float16, torch.bfloat16, torch.float32):
                for dim in (129, 257, 512):
                    with self.subTest(dtype=dtype, output_dtype=output_dtype, dim=dim):
                        x = torch.randn(3, 5, dim, device="npu", dtype=dtype)
                        self.check_output(x, output_dtype=output_dtype)

    def test_zero_and_near_zero(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for dim in (256, 512):
                for scale in (0.0, 1e-7):
                    for eps in (1e-6, 1e-3):
                        with self.subTest(dtype=dtype, dim=dim, scale=scale, eps=eps):
                            x = (torch.randn(17, dim) * scale).to(
                                device="npu", dtype=dtype
                            )
                            self.check_output(x, eps=eps)


if __name__ == "__main__":
    unittest.main()
