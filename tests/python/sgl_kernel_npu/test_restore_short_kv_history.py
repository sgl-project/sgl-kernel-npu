import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    free_shm,
    restore_short_kv_history_inplace,
)


class TestRestoreShortKVHistory(unittest.TestCase):
    def test_graph_replay_and_bounds(self):
        device = torch.npu.current_device()
        host, _, ptr = create_shm_tensor((1, 8, 1, 16), torch.float16, device)
        host.copy_(torch.arange(host.numel(), dtype=host.dtype).reshape(host.shape))
        cache = torch.full((4, 8, 1, 16), -1, dtype=host.dtype, device="npu")
        table = torch.tensor(
            [
                [0] * 10,
                [0, 1, -1, 8, 4, 5, 6, 7, 0, 0],
                [7, 6, 5, 4, 3, 2, 1, 0, 0, 0],
                [0] * 10,
            ],
            dtype=torch.int32,
            device="npu",
        )
        requests = torch.tensor([1, 2, 0], dtype=torch.int64, device="npu")
        lengths = torch.tensor([6, 8, 0], dtype=torch.int32, device="npu")
        first_storage = torch.tensor(
            [[5, 99], [4, 99], [0, 99]], dtype=torch.int64, device="npu"
        )
        first = first_storage[:, 0]
        old_begin = torch.tensor([-1, 0, -1, -1], dtype=torch.int32, device="npu")
        old_end = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="npu")
        next_begin = torch.full((4,), -9, dtype=torch.int32, device="npu")
        next_end = next_begin.clone()
        next_first = next_begin.clone()

        def run():
            restore_short_kv_history_inplace(
                host,
                cache,
                table,
                requests,
                lengths,
                first,
                old_begin,
                old_end,
                next_begin,
                next_end,
                next_first,
                8,
                8,
                src_ptr=ptr,
            )

        try:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
            graph.replay()
            torch.npu.synchronize()
            expected = torch.full((4, 8, 1, 16), -1, dtype=host.dtype)
            expected[1, 1] = host[0, 1]
            expected[1, 4] = host[0, 4]
            self.assertTrue(torch.equal(cache.cpu(), expected))
            self.assertEqual(next_begin.cpu().tolist(), [-9, 0, -1, -9])
            self.assertEqual(next_end.cpu().tolist(), [-9, 6, 8, -9])
            self.assertEqual(next_first.cpu().tolist(), [-9, 5, 4, -9])

            requests.copy_(torch.tensor([2, 1, 0], dtype=torch.int64, device="npu"))
            lengths.copy_(torch.tensor([5, 8, 0], dtype=torch.int32, device="npu"))
            first_storage.copy_(
                torch.tensor(
                    [[3, 99], [4, 99], [0, 99]], dtype=torch.int64, device="npu"
                )
            )
            graph.replay()
            torch.npu.synchronize()
            expected[2, :3] = host[0, torch.tensor([7, 6, 5])]
            self.assertTrue(torch.equal(cache.cpu(), expected))
            self.assertEqual(next_begin.cpu().tolist(), [-9, -1, 0, -9])
            self.assertEqual(next_end.cpu().tolist(), [-9, 8, 5, -9])
            self.assertEqual(next_first.cpu().tolist(), [-9, 4, 3, -9])

            # Invalid real requests clear their next extents; padding stays untouched.
            for invalid_length, invalid_first in ((0, 0), (9, 0), (5, -1), (5, 6)):
                requests.copy_(
                    torch.tensor([0, 3, -1], dtype=torch.int64, device="npu")
                )
                lengths.copy_(
                    torch.tensor(
                        [5, invalid_length, 0], dtype=torch.int32, device="npu"
                    )
                )
                first_storage.copy_(
                    torch.tensor(
                        [[1, 99], [invalid_first, 99], [0, 99]],
                        dtype=torch.int64,
                        device="npu",
                    )
                )
                next_begin.fill_(-9)
                next_end.fill_(-9)
                next_first.fill_(-9)
                graph.replay()
                torch.npu.synchronize()
                self.assertTrue(torch.equal(cache.cpu(), expected))
                self.assertEqual(next_begin.cpu().tolist(), [-9, -9, -9, -1])
                self.assertEqual(next_end.cpu().tolist(), [-9, -9, -9, -1])
                self.assertEqual(next_first.cpu().tolist(), [-9, -9, -9, -1])

            with self.assertRaisesRegex(
                RuntimeError, "extent buffers must not overlap"
            ):
                restore_short_kv_history_inplace(
                    host,
                    cache,
                    table,
                    requests,
                    lengths,
                    first,
                    old_begin,
                    old_end,
                    old_begin,
                    next_end,
                    next_first,
                    8,
                    8,
                    src_ptr=ptr,
                )
        finally:
            torch.npu.synchronize()
            free_shm(device)


if __name__ == "__main__":
    unittest.main()
