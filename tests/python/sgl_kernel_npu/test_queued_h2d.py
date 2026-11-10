# SPDX-License-Identifier: BSD-3-Clause
"""Port the measured prototype's byte, layer-slice and queue-lifetime cases.

Run in separate processes with TASK_QUEUE_ENABLE=0 and TASK_QUEUE_ENABLE=2.
Historical prototype results do not constitute a run of this integration.
"""

import unittest

import torch
from sgl_kernel_npu._queued_h2d import queued_h2d_copy


class TestQueuedH2D(unittest.TestCase):
    page_size = 128
    specs = (
        (78, 656, torch.int8),
        (21, 128, torch.bfloat16),
        (1, 512, torch.bfloat16),
        (1, 64, torch.bfloat16),
        (1, 128, torch.bfloat16),
    )

    def indices(self, pages):
        return (
            torch.tensor(pages, dtype=torch.int64)[:, None] * self.page_size
            + torch.arange(self.page_size)
        ).flatten()

    def make(self, pages):
        parts = []
        for i, (layers, width, dtype) in enumerate(self.specs):
            host = torch.empty(
                (pages, layers, self.page_size, 1, width),
                dtype=dtype,
                pin_memory=True,
            ).fill_(11 + i)
            pattern = (
                torch.arange(pages)[:, None] * 3 + torch.arange(layers)[None, :]
            ) % 97 + 1
            host[:, :, 0, 0, 0] = pattern.to(dtype)
            host[:, :, -1, 0, -1] = (pattern + 1).to(dtype)
            device = torch.zeros(
                (layers, pages, self.page_size, 1, width), dtype=dtype, device="npu"
            )
            parts.append((device, host))
        return parts

    def test_strided_layers_preserve_all_other_bytes(self):
        host_pages = [2, 3, 4, 7, 10, 11, 19, 25]
        device_pages = [0, 1, 2, 4, 7, 8, 20, 31]
        parts = self.make(32)
        views = [
            (
                device[1:2] if device.shape[0] > 1 else device,
                host[:, 1:2] if host.shape[1] > 1 else host,
            )
            for device, host in parts
        ]
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            queued_h2d_copy(
                views,
                self.indices(host_pages),
                self.indices(device_pages),
                self.page_size,
            )
        stream.synchronize()
        for device, host in parts:
            actual = device.cpu()
            expected = torch.zeros_like(actual)
            layer = 1 if device.shape[0] > 1 else 0
            for h, d in zip(host_pages, device_pages):
                expected[layer, d] = host[h, layer]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_six_generations_keep_host_lifetime_and_stream_order(self):
        stream = torch.npu.Stream()
        observations = []
        with torch.npu.stream(stream):
            for generation in range(6):
                parts = self.make(16)
                for _, host in parts:
                    host.fill_(generation + 21)
                queued_h2d_copy(
                    parts,
                    self.indices([0, 1, 4, 5, 8, 9]),
                    self.indices([1, 2, 6, 7, 12, 13]),
                    self.page_size,
                )
                observations.append(
                    (generation, [d[0, 1, 0, 0, :16].clone() for d, _ in parts])
                )
                del parts
        stream.synchronize()
        for generation, values in observations:
            for value in values:
                self.assertTrue(bool((value.cpu() == generation + 21).all()))

    def test_invalid_complete_page_indices_are_rejected(self):
        parts = self.make(16)
        host_indices = self.indices([0, 1])
        device_indices = self.indices([1, 2])
        for position, value in ((0, -128), (2, 131)):
            bad = device_indices.clone()
            bad[position] = value
            with self.assertRaises(RuntimeError):
                queued_h2d_copy(parts, host_indices, bad, self.page_size)


if __name__ == "__main__":
    unittest.main()
