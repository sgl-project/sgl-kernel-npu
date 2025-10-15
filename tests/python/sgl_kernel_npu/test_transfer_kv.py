import random
import time
import unittest

import sgl_kernel_npu
import torch
import torch_npu

# example comes from Qwen3-32B, TP=2
TP=2
NUM_KV_HEADS = 8
NUM_LAYERS = 64
NUM_PAGES = 30
PAGE_SIZE = 128
HEAD_NUM_PER_TP = int(NUM_KV_HEADS / TP)
HEAD_DIM = 128

H2D=1
D2H=2

class TestTransferKV(unittest.TestCase):

    def _kv_transfer(self, kind):
        torch.npu.set_device(0)

        device_kv_buffer = torch.ones(
            (2, NUM_LAYERS, NUM_PAGES, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="npu",
        )
        device_k = device_kv_buffer[0]
        device_v = device_kv_buffer[1]

        host_kv_buffer = torch.zeros(
            (2, NUM_PAGES, NUM_LAYERS, PAGE_SIZE, HEAD_NUM_PER_TP, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cpu",
        )

        self.assertNotEqual(device_kv_buffer.sum(), host_kv_buffer.sum(),
                            "device value should not be equal to host value")

        host_k = host_kv_buffer[0]
        host_v = host_kv_buffer[1]

        device_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(NUM_PAGES * PAGE_SIZE, dtype=torch.int64)

        stream = torch.npu.Stream()
        finish_event = torch.npu.Event()
        start = time.time()
        with torch.npu.stream(stream):
            torch.ops.npu.transfer_kv(device_k, host_k, device_v, host_v,
                                      device_indices, host_indices, kind,
                                      0, NUM_LAYERS, PAGE_SIZE)
            finish_event.record()
        finish_event.synchronize()

        end = time.time()
        kind_str = "D2H" if kind == 2 else "H2D"
        print(f"kv transfer {kind_str}, "
              f"tensor copy times is {NUM_PAGES * NUM_LAYERS * 2}, "
              f"single tensor copy size is {PAGE_SIZE * HEAD_NUM_PER_TP * HEAD_DIM * torch.bfloat16.itemsize} bytes, "
              f"total duration {float((end - start) * 1000):.3f}ms")

        return device_kv_buffer, host_kv_buffer

    def test_page_copy_d2h(self):
        device_kv, host_kv = self._kv_transfer(D2H)

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            host_kv.sum().item(),
            delta=1e-3,
            msg="host value should be equal to device value after transfer d2h"
        )

        self.assertAlmostEqual(
            host_kv.sum().item(),
            host_kv.numel(),
            delta=1e-3,
            msg="host value sum() should be equal to numel() after transfer d2h"
        )

    def test_page_copy_h2d(self):
        device_kv, host_kv = self._kv_transfer(H2D)

        self.assertAlmostEqual(
            host_kv.sum().item(),
            device_kv.sum().cpu().item(),
            delta=1e-3,
            msg="device value should be equal to host value after transfer h2d")

        self.assertAlmostEqual(
            device_kv.sum().cpu().item(),
            0,
            delta=1e-3,
            msg="device value sum() should be equal to 0 after transfer h2d")


if __name__ == '__main__':
    unittest.main()

