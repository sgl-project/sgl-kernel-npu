import unittest

import sgl_kernel_npu  # noqa: F401
import torch


class TestShmAllocator(unittest.TestCase):
    def test_create_dispatches_without_tensor_arguments(self):
        with self.assertRaisesRegex(RuntimeError, "size must be positive"):
            torch.ops.npu.shm_allocator_create_and_register(0, 0, "")

    def test_free_dispatches_without_tensor_arguments(self):
        with self.assertRaisesRegex(RuntimeError, "device_id must be non-negative"):
            torch.ops.npu.shm_allocator_free_all(-1)


if __name__ == "__main__":
    unittest.main()
