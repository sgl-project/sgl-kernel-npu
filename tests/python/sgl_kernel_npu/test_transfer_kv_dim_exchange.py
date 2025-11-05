import time
import unittest

import torch
from sgl_kernel_npu.kvcacheio import (
    TransferDirection,
    TransferFlag,
    transfer_kv_dim_exchange,
)

# example comes from Qwen3-32B, TP=2
TP = 2
NUM_KV_HEADS = 8
NUM_LAYERS = 64
NUM_PAGES = 30
PAGE_SIZE = 128
HEAD_NUM_PER_TP = int(NUM_KV_HEADS / TP)
HEAD_DIM = 128


class TestTransferKV(unittest.TestCase):

    def _kv_transfer(
        self, direct: TransferDirection, v_empty: bool, index_k_empty: bool = True
    ):
        torch.npu.set_device(0)

        device_kv_buffer = torch.ones(
            (2, NUM_LAYERS, NUM_PAGES, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="npu",
        )
        device_k = device_kv_buffer[0]
        device_v = torch.empty(0) if v_empty else device_kv_buffer[1]

        host_kv_buffer = torch.zeros(
            (2, NUM_PAGES, NUM_LAYERS, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cpu",
        )

        self.assertNotEqual(
            device_kv_buffer.sum(),
            host_kv_buffer.sum(),
            "device value should not be equal to host value",
        )

        host_k = host_kv_buffer[0]
        host_v = torch.empty(0) if v_empty else host_kv_buffer[1]

        device_index_k = torch.ones(
            (NUM_PAGES, NUM_LAYERS, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="npu",
        )

        host_index_k = torch.zeros(
            (NUM_PAGES, NUM_LAYERS, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cpu",
        )

        device_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)

        stream = torch.npu.Stream()
        start = time.time()
        with torch.npu.stream(stream):
            if index_k_empty:
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_k,
                    host_k=host_k,
                    device_v=device_v,
                    host_v=host_v,
                    page_size=PAGE_SIZE,
                    direction=direct.value,
                    flags=TransferFlag.value,
                )
            else:
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_k,
                    host_k=host_k,
                    device_v=device_v,
                    host_v=host_v,
                    device_index_k=device_index_k,
                    host_index_k=host_index_k,
                    page_size=PAGE_SIZE,
                    direction=direct.value,
                    flags=TransferFlag.value,
                )

        end = time.time()
        direct_str = "D2H" if direct.value == 2 else "H2D"
        copy_times = NUM_PAGES
        if v_empty is False:
            copy_times += NUM_PAGES
        if index_k_empty is False:
            copy_times += NUM_PAGES

        total_size = (
            copy_times
            * NUM_LAYERS
            * PAGE_SIZE
            * HEAD_NUM_PER_TP
            * HEAD_DIM
            * torch.bfloat16.itemsize
        )
        print(
            f"kv transfer {direct_str}, {v_empty=}, {index_k_empty=}, "
            f"2d copy times is {copy_times}, "
            f"total copy size is {total_size} bytes, "
            f"total duration {float((end - start) * 1000):.3f}ms"
        )

        return device_kv_buffer, host_kv_buffer

    def _k_transfer(self, direct_str):
        return self._kv_transfer(direct_str, True)

    def test_kv_copy_d2h(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.D2H, False)

        self.assertAlmostEqual(
            host_kv.sum().item(),
            device_kv.sum().cpu().item(),
            delta=1e-3,
            msg="host value should be equal to device value after transfer kv d2h",
        )

        self.assertAlmostEqual(
            host_kv.sum().item(),
            host_kv.numel(),
            delta=1e-3,
            msg="host value sum() should be equal to numel() after transfer kv d2h",
        )

    def test_kv_copy_h2d(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.H2D, False)

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            host_kv.sum().item(),
            delta=1e-3,
            msg="device value should be equal to host value after transfer kv h2d",
        )

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            0,
            delta=1e-3,
            msg="device value sum() should be equal to 0 after transfer kv h2d",
        )

    def test_kv_index_k_copy_d2h(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.D2H, False, False)

        self.assertAlmostEqual(
            host_kv.sum().item(),
            device_kv.sum().cpu().item(),
            delta=1e-3,
            msg="host value should be equal to device value after transfer kv and index k d2h",
        )

        self.assertAlmostEqual(
            host_kv.sum().item(),
            host_kv.numel(),
            delta=1e-3,
            msg="host value sum() should be equal to numel() after transfer kv and index d2h",
        )

    def test_kv_index_k_copy_h2d(self):
        device_kv, host_kv = self._kv_transfer(TransferDirection.H2D, False, False)

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            host_kv.sum().item(),
            delta=1e-3,
            msg="device value should be equal to host value after transfer kv and index k h2d",
        )

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            0,
            delta=1e-3,
            msg="device value sum() should be equal to 0 after transfer kv and index k h2d",
        )

    def test_k_copy_d2h(self):
        device_kv, host_kv = self._k_transfer(TransferDirection.D2H)

        self.assertAlmostEqual(
            host_kv.sum().item() * 2,
            device_kv.sum().cpu().item(),
            delta=1e-3,
            msg="host value * 2 should be equal to device value after transfer k d2h",
        )

        self.assertAlmostEqual(
            host_kv.sum().item() * 2,
            host_kv.numel(),
            delta=1e-3,
            msg="host value sum() * 2 should be equal to numel() after transfer k d2h",
        )

    def test_k_copy_h2d(self):
        device_kv, host_kv = self._k_transfer(TransferDirection.H2D)

        self.assertAlmostEqual(
            device_kv[0].sum().cpu().item(),
            0,
            delta=1e-3,
            msg="device k sum() should be equal to 0 after transfer k h2d",
        )

        self.assertAlmostEqual(
            device_kv[1].sum().cpu().item() * 2,
            host_kv.numel(),
            delta=1e-3,
            msg="device v sum() * 2 should be equal to host value after transfer k h2d",
        )


if __name__ == "__main__":
    unittest.main()
