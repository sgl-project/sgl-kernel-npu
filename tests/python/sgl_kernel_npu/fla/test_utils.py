import unittest
from unittest.mock import patch

import torch
from sgl_kernel_npu.fla.utils import input_guard


class TestUtils(unittest.TestCase):

    @patch('sgl_kernel_npu.fla.utils.custom_device_ctx')
    def test_input_guard_with_enable_torch_compile_true(self, mock_custom_device_ctx):
        @input_guard
        def f(x):
            return x + 1

        tensor = torch.randn(2, 2)
        f(tensor, enable_torch_compile=True)
        mock_custom_device_ctx.assert_not_called()

    @patch('sgl_kernel_npu.fla.utils.custom_device_ctx')
    def test_input_guard_with_enable_torch_compile_false(self, mock_custom_device_ctx):
        @input_guard
        def f(x):
            return x + 1

        tensor = torch.randn(2, 2)
        f(tensor)
        mock_custom_device_ctx.assert_called_once()


if __name__ == "__main__":
    unittest.main()
