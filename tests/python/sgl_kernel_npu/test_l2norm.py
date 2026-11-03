import unittest
from unittest.mock import patch

import torch
import torch_npu
from sgl_kernel_npu.fla.l2norm import l2norm, l2norm_fwd_kernel_opt
from sgl_kernel_npu.utils.triton_utils import get_device_properties


class TestL2Norm(unittest.TestCase):
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

    def test_shapes_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for dim in (1, 127, 128, 513, 1024):
                for rows in (1, 17, 129, 2053):
                    with self.subTest(dtype=dtype, dim=dim, rows=rows):
                        x = torch.randn(rows, dim, device="npu", dtype=dtype)
                        self.check_output(x)

    def test_epsilon_and_output_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for output_dtype in (torch.float16, torch.bfloat16, torch.float32):
                for dim in (128, 513):
                    with self.subTest(dtype=dtype, output_dtype=output_dtype, dim=dim):
                        x = torch.randn(2, 3, dim, device="npu", dtype=dtype)
                        for eps in (0.0, 1e-6, 1e-3):
                            self.check_output(x, eps=eps, output_dtype=output_dtype)

    def test_zero_and_near_zero(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for dim in (128, 513):
                for scale in (0.0, 1e-7):
                    for eps in (1e-6, 1e-3):
                        with self.subTest(dtype=dtype, dim=dim, scale=scale, eps=eps):
                            x = (torch.randn(17, dim) * scale).to(
                                device="npu", dtype=dtype
                            )
                            self.check_output(x, eps=eps)

    def test_runtime_sizes_reuse_compiled_kernel(self):
        # Keep dtype, feature dimension and pointer alignment fixed. These row
        # counts vary equality-to-one and divisibility specializations of T,
        # NB=ceil(T/2048), and MBS=ceil(T/num_cores), including multiple BT tiles.
        num_cores = get_device_properties()[1]
        rows = (
            1,
            3,
            16,
            17,
            128,
            129,
            2048,
            2049,
            4097,
            num_cores * 109 + 1,
            num_cores * 128,
        )
        run = l2norm_fwd_kernel_opt.run
        compiled = []

        def record_kernel(*args, **kwargs):
            kernel = run(*args, **kwargs)
            compiled.append(kernel)
            return kernel

        with patch.object(l2norm_fwd_kernel_opt, "run", side_effect=record_kernel):
            for n in rows:
                x = torch.randn(n, 128, device="npu", dtype=torch.bfloat16)
                self.check_output(x)

        self.assertEqual(len(compiled), len(rows))
        self.assertIsNotNone(compiled[0])
        for n, kernel in zip(rows, compiled):
            with self.subTest(rows=n):
                self.assertIs(kernel, compiled[0])


if __name__ == "__main__":
    unittest.main()
