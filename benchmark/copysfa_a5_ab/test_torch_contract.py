"""CPU Torch checks for the actual LIM-to-CopySFA benchmark wiring."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from fixture import FixtureSpec, _index_key_scales, build_torch_fixture  # noqa: E402


class LimCopyContractTest(unittest.TestCase):
    def test_long_context_topk_boundary_survives_bf16_sorting(self):
        for width in (1, 4):
            spec = FixtureSpec(width, 1, 8, 64, 16000, 139904, 140032)
            scores = _index_key_scales(torch, spec, spec.context).flatten() * 128
            for dtype in (torch.float32, torch.bfloat16):
                rounded = scores.to(dtype).float()
                for route in range(width):
                    with self.subTest(width=width, dtype=dtype, route=route):
                        visible = spec.context - (width - 1 - route)
                        chosen = (
                            torch.topk(rounded[:visible], 2048).indices.sort().values
                        )
                        expected = torch.tensor(spec.route_ids(route))
                        self.assertTrue(torch.equal(chosen, expected))
                        self.assertGreater(
                            float(rounded[visible - 2048]),
                            float(rounded[visible - 2049]),
                        )

    def test_lim_output_views_reach_copy_without_conversion_or_copy(self):
        for width in (1, 4):
            with self.subTest(width=width):
                fixture = build_torch_fixture(
                    torch, FixtureSpec(width, 1, 8, 32, 8192, 9216, 9345)
                )
                seen = {}

                def lim(*args):
                    self.assertEqual(tuple(args[16].shape), (1, 32768))
                    self.assertEqual(tuple(args[17].shape), (1, 32768))
                    self.assertTrue(args[16].is_contiguous())
                    self.assertTrue(args[17].is_contiguous())
                    seen["sources"], seen["slots"] = args[16:18]
                    args[16].fill_(123)
                    args[17].fill_(456)

                def copy(*args):
                    self.assertEqual(tuple(args[12].shape), (1, 1, 32768))
                    self.assertEqual(tuple(args[13].shape), (1, 1, 32768))
                    self.assertEqual(args[12].data_ptr(), seen["sources"].data_ptr())
                    self.assertEqual(args[13].data_ptr(), seen["slots"].data_ptr())
                    self.assertTrue(bool((args[12] == 123).all()))
                    self.assertTrue(bool((args[13] == 456).all()))
                    seen["copied"] = True

                def create(shape, dtype, device_id):
                    return torch.empty(shape, dtype=dtype), 0, 1

                with (
                    patch.object(
                        bench, "_import_shm", return_value=(create, lambda: None)
                    ),
                    patch.object(
                        torch.ops.npu,
                        "fused_quant_lightning_indexer_manage",
                        types.SimpleNamespace(default=lim),
                        create=True,
                    ),
                    patch.object(
                        torch.ops.npu,
                        "fused_quant_scatter_copy_sparse_flash_attention",
                        types.SimpleNamespace(default=copy),
                        create=True,
                    ),
                ):
                    runtime = bench._build_lim_copysfa(
                        torch, fixture, torch.device("cpu"), 0
                    )
                    runtime["chain"]()
                    self.assertTrue(seen["copied"])


if __name__ == "__main__":
    unittest.main()
