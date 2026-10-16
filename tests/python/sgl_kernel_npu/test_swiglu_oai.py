import unittest

import torch


def reference_swiglu_oai(hidden_states, dim, alpha, limit):
    """Independent CPU reference with one final conversion to the output dtype."""
    x = hidden_states.detach().cpu().double().reshape(-1, dim)
    gate = x[:, 0::2].clamp(max=limit)
    up = x[:, 1::2].clamp(min=-limit, max=limit)
    return (up + 1) * gate * torch.sigmoid(alpha * gate)


def assert_swiglu_close(actual, expected, dtype):
    actual = actual.detach().cpu()
    rounded = expected.to(dtype)
    # Allow one representable step around the correctly rounded FP64 result.
    # There is no absolute tolerance that could hide an error near zero.
    upper = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
    lower = torch.nextafter(rounded, torch.full_like(rounded, -float("inf")))
    if not torch.isfinite(actual).all():
        raise AssertionError("non-finite SwiGLU output")
    bad = (actual < lower) | (actual > upper)
    bad |= (rounded == 0) & (actual != 0)
    if bad.any():
        difference = (actual.double() - expected).abs()
        raise AssertionError(
            f"{bad.sum().item()} values exceed one output ULP; "
            f"max_abs={difference.max().item():.9g}"
        )


class TestSwigluOAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu
        from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai_triton

        torch_npu.npu.set_device(0)
        cls.kernel = staticmethod(swiglu_oai_triton)

    def make_input(self, shape, dtype):
        generator = torch.Generator().manual_seed(20261013)
        return (torch.rand(shape, generator=generator) * 40 - 20).to(dtype)

    def check_case(self, x, dim, alpha=1.702, limit=7.0):
        before = x.cpu().clone()
        output = self.kernel(x, dim, alpha, limit)
        torch.npu.synchronize()
        self.assertEqual(output.shape, (x.numel() // dim, dim // 2))
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.device, x.device)
        self.assertTrue(output.is_contiguous())
        torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)
        reference = reference_swiglu_oai(before, dim, alpha, limit)
        assert_swiglu_close(output, reference, x.dtype)

    def test_gpt_oss_single_row(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_case(self.make_input((1, 5760), dtype).npu(), 5760)

    def test_gpt_oss_intermediate_width(self):
        # GPT-OSS routes tokens to experts with 2880 intermediate elements.
        # Even a single row used to allocate a 16-by-2880 activation tile.
        for dtype in (torch.float16, torch.bfloat16):
            for rows in (17, 65, 257):
                with self.subTest(dtype=dtype, rows=rows):
                    self.check_case(self.make_input((rows, 5760), dtype).npu(), 5760)

    def test_tile_tails_and_wide_rows(self):
        for dtype in (torch.float16, torch.bfloat16):
            for rows, dim in ((1, 2), (3, 34), (17, 1022), (3, 2050), (3, 32768)):
                with self.subTest(dtype=dtype, rows=rows, dim=dim):
                    self.check_case(self.make_input((rows, dim), dtype).npu(), dim)

    def test_alpha_and_clamp_boundaries(self):
        values = torch.tensor([-40, -10, -7, -1, 0, 1, 7, 10, 40])
        pairs = torch.cartesian_prod(values, values).flatten().view(1, -1)
        for dtype in (torch.float16, torch.bfloat16):
            for alpha, limit in ((1.702, 7.0), (0.5, 1.0), (2.0, 10.0), (0.0, 0.0)):
                with self.subTest(dtype=dtype, alpha=alpha, limit=limit):
                    self.check_case(pairs.to(dtype).npu(), pairs.numel(), alpha, limit)

    def test_strided_input(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype, layout="sliced"):
                base = self.make_input((6, 4100), dtype).npu()
                self.check_case(base[::2, 1::2], 2050)
            with self.subTest(dtype=dtype, layout="transposed"):
                base = self.make_input((2050, 3), dtype).npu()
                self.check_case(base.t(), 2050)

    def test_flattened_and_empty_input(self):
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((2050,), (2, 3, 2050), (0, 5760)):
                with self.subTest(dtype=dtype, shape=shape):
                    self.check_case(self.make_input(shape, dtype).npu(), shape[-1])

    def test_invalid_dim(self):
        x = torch.empty((1, 6), device="npu", dtype=torch.bfloat16)
        for dim in (0, -2, 3):
            with self.subTest(dim=dim):
                with self.assertRaisesRegex(ValueError, "positive even"):
                    self.kernel(x, dim, 1.702, 7.0)


if __name__ == "__main__":
    unittest.main()
