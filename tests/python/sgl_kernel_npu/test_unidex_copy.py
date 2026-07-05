import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mem_cache import unidex_copy_inplace


class TestUnidexCopy(unittest.TestCase):
    def _run_copy(self, block_elements, raw_mode=None):
        src = torch.arange(
            8 * block_elements, dtype=torch.float16, device="npu"
        ).reshape(2, 2, 2, block_elements)
        dst = torch.full((2, 2, block_elements), -1, dtype=torch.float16, device="npu")
        src_index = torch.tensor([0, 3, 5, 7], dtype=torch.int64, device="npu")
        dst_index = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="npu")
        valid_mask = torch.tensor([True, False, True, True], device="npu")

        expected = dst.clone().reshape(4, block_elements)
        expected[0] = src.reshape(8, block_elements)[0]
        expected[2] = src.reshape(8, block_elements)[5]
        expected[3] = src.reshape(8, block_elements)[7]

        pointer_args = {}
        if raw_mode in ("src", "both"):
            pointer_args["src_ptr"] = src.data_ptr()
        if raw_mode in ("dst", "both"):
            pointer_args["dst_ptr"] = dst.data_ptr()
        result = unidex_copy_inplace(
            src,
            dst,
            src_index,
            dst_index,
            valid_mask,
            src_address_ndims=3,
            dst_address_ndims=2,
            **pointer_args,
        )
        torch.npu.synchronize()
        self.assertIs(result, dst)
        self.assertTrue(torch.equal(dst.reshape(4, block_elements), expected))

    def test_aligned_rows(self):
        self._run_copy(block_elements=32)

    def test_unaligned_rows(self):
        self._run_copy(block_elements=15)

    def test_raw_source_pointer(self):
        self._run_copy(block_elements=32, raw_mode="src")

    def test_raw_destination_pointer(self):
        self._run_copy(block_elements=32, raw_mode="dst")

    def test_raw_source_and_destination_pointers(self):
        self._run_copy(block_elements=32, raw_mode="both")

    def test_empty_mapping_is_noop(self):
        src = torch.arange(16, dtype=torch.float16, device="npu").reshape(2, 8)
        dst = torch.full((2, 8), -1, dtype=torch.float16, device="npu")
        before = dst.clone()
        empty_index = torch.empty(0, dtype=torch.int64, device="npu")
        empty_mask = torch.empty(0, dtype=torch.bool, device="npu")

        unidex_copy_inplace(
            src,
            dst,
            empty_index,
            empty_index,
            empty_mask,
            src_address_ndims=1,
            dst_address_ndims=1,
        )
        self.assertTrue(torch.equal(dst, before))

    def test_rejects_wrong_index_dtype(self):
        src = torch.zeros((2, 8), dtype=torch.float16, device="npu")
        dst = torch.zeros_like(src)
        index = torch.tensor([0], dtype=torch.int32, device="npu")
        mask = torch.tensor([True], device="npu")

        with self.assertRaisesRegex(RuntimeError, "src_index must be int64"):
            unidex_copy_inplace(
                src,
                dst,
                index,
                index,
                mask,
                src_address_ndims=1,
                dst_address_ndims=1,
            )


if __name__ == "__main__":
    unittest.main()
