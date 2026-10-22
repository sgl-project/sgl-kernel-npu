"""NPU strided Adler-32 checks against Python's independent zlib oracle."""

import unittest
import zlib

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mem_cache.checksum import adler32_strided_checksum


def reference(tensors, strides, indices):
    value = 1
    for tensor, stride, idx in zip(tensors, strides, indices):
        raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        for row in idx.cpu().tolist():
            value = zlib.adler32(raw[row * stride : (row + 1) * stride], value)
    return value


class TestAdler32StridedChecksum(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        torch.manual_seed(42)

    def check_descriptors(self, tensors, strides, indices):
        actual = adler32_strided_checksum(
            [t.data_ptr() for t in tensors], strides, indices
        )
        self.assertEqual(actual, reference(tensors, strides, indices))
        return actual

    def test_empty(self):
        self.assertEqual(adler32_strided_checksum([], [], []), 1)
        empty = torch.empty(0, dtype=torch.int64, device=self.device)
        self.assertEqual(adler32_strided_checksum([0], [64], [empty]), 1)
        idx = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.assertEqual(adler32_strided_checksum([0], [0], [idx]), 1)

    def test_byte_tails_order_and_duplicates(self):
        for stride in (1, 31, 1023, 1024, 1025, 65521, 70001):
            with self.subTest(stride=stride):
                data = torch.randint(0, 256, (8, stride), dtype=torch.uint8).to(
                    self.device
                )
                idx = torch.tensor(
                    [7, 0, 3, 3, 1], dtype=torch.int64, device=self.device
                )
                self.check_descriptors([data], [stride], [idx])

    def test_multiple_dtypes_and_empty_component(self):
        tensors = [
            torch.randn(9, 35, dtype=dtype).to(self.device)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ]
        indices = [
            torch.tensor(rows, dtype=torch.int32, device=self.device)
            for rows in ([8, 1, 4], [], [2, 0])
        ]
        self.check_descriptors(tensors, [t[0].nbytes for t in tensors], indices)

    def test_modular_overflow(self):
        data = torch.full((257, 4097), 255, dtype=torch.uint8, device=self.device)
        idx = torch.arange(256, -1, -1, dtype=torch.int64, device=self.device)
        self.check_descriptors([data], [4097], [idx])

    def test_selection_and_corruption(self):
        data = torch.arange(256, dtype=torch.int32).view(8, 32).to(self.device)
        idx = torch.tensor([5, 1], dtype=torch.int64, device=self.device)
        first = self.check_descriptors([data], [128], [idx])
        data[0, 0] += 1
        self.assertEqual(first, self.check_descriptors([data], [128], [idx]))
        data[5, 0] += 1
        self.assertNotEqual(first, self.check_descriptors([data], [128], [idx]))
        self.assertNotEqual(
            self.check_descriptors([data], [128], [idx]),
            self.check_descriptors([data], [128], [idx.flip(0)]),
        )

    def test_invalid_descriptors(self):
        idx = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        data = torch.zeros((2, 64), dtype=torch.uint8, device=self.device)
        with self.assertRaises(ValueError):
            adler32_strided_checksum([data.data_ptr()], [], [idx])
        for invalid in (idx.view(1, 2), idx.to(torch.float32), idx.cpu()):
            with self.assertRaises(ValueError):
                adler32_strided_checksum([data.data_ptr()], [64], [invalid])


if __name__ == "__main__":
    unittest.main()
