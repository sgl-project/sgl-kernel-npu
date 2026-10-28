"""NPU availability and exact output checks for sampling tests."""

import importlib.util
import unittest

import torch

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available()
else:
    NPU_AVAILABLE = False


class SamplingTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def assert_output(self, actual, expected):
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
