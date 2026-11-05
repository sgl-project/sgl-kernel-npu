"""CPU checks for the indexer reference and profile; no installed NPU wheel needed."""

import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load(
    "indexer_reference",
    PACKAGE / "implementation/torch_native/attention/lightning_indexer.py",
)
validation = load(
    "indexer_validation", PACKAGE / "operators/attention/_lightning_indexer.py"
)


class TestIndexerReference(unittest.TestCase):
    def setUp(self):
        self.q = torch.ones(1, 2, 1, 128, dtype=torch.float16)
        self.k = torch.zeros(1, 8, 1, 128, dtype=torch.float16)
        self.k[0, :, 0, 0] = torch.arange(8)
        self.w = torch.ones(1, 2, 1, dtype=torch.float32)

    def test_dense_scores_and_padding(self):
        indices, values = native.lightning_indexer_v2(
            self.q, self.k, self.w, 10, return_value=1
        )
        self.assertEqual(indices[0, 0, 0].tolist(), [7, 6, 5, 4, 3, 2, 1, 0, -1, -1])
        self.assertEqual(values[0, 0, 0, :8].tolist(), list(range(7, -1, -1)))
        self.assertTrue(torch.isneginf(values[..., -2:]).all())

    def test_relu_precedes_weighted_head_reduction(self):
        q = torch.zeros(1, 1, 2, 128, dtype=torch.float16)
        q[0, 0, :, 0] = torch.tensor([1, -1], dtype=torch.float16)
        k = torch.zeros(1, 2, 1, 128, dtype=torch.float16)
        k[0, :, 0, 0] = torch.tensor([2, -1], dtype=torch.float16)
        w = torch.tensor([[[2, -3]]], dtype=torch.float32)
        indices, values = native.lightning_indexer_v2(q, k, w, 2, return_value=1)
        self.assertEqual(indices.flatten().tolist(), [0, 1])
        self.assertEqual(values.flatten().tolist(), [4, -3])

    def test_causal_and_partial_lengths(self):
        indices, _ = native.lightning_indexer_v2(self.q, self.k, self.w, 8, mask_mode=3)
        self.assertEqual(indices[0, 0, 0, 0].item(), 6)
        self.assertEqual(indices[0, 0, 0, -1].item(), -1)
        indices, _ = native.lightning_indexer_v2(
            self.q,
            self.k,
            self.w,
            8,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            seqused_k=torch.tensor([3], dtype=torch.int32),
        )
        self.assertTrue((indices[0, 1] == -1).all())
        self.assertEqual(indices[0, 0, 0, :3].tolist(), [2, 1, 0])

    def test_source_pins_last_block(self):
        self.k[0, 0, 0, 0] = 100
        out = native.lightning_indexer_v2(
            self.q, self.k, self.w, 1, candidate_topk_blocks=1, candidate_block_size=2
        )
        self.assertEqual(out[2][0, 0, 0].tolist(), [3])
        self.assertEqual(out[3].numel(), 0)

    def test_source_arity_and_empty_rows(self):
        self.assertEqual(len(native.lightning_indexer_v2(self.q, self.k, self.w, 1)), 2)
        out = native.lightning_indexer_v2(
            self.q, self.k, self.w, 1, candidate_topk_blocks=-1
        )
        self.assertEqual(len(out), 4)
        self.assertEqual(out[2].numel(), 0)
        out = native.lightning_indexer_v2(
            self.q,
            self.k,
            self.w,
            1,
            candidate_topk_blocks=64,
            seqused_k=torch.tensor([0], dtype=torch.int32),
        )
        self.assertTrue((out[2] == -1).all())

    def test_sparse_no_leak_dirty_candidates(self):
        candidates = torch.tensor([0, 0, -1, 999], dtype=torch.int32).expand(1, 2, 1, 4)
        indices, values = native.sparse_lightning_indexer(
            self.q,
            self.k,
            self.w,
            4,
            candidates,
            candidate_block_size=2,
            return_value=1,
        )
        self.assertEqual(indices[0, 0, 0].tolist(), [1, 0, -1, -1])
        self.assertEqual(values[0, 0, 0].tolist(), [1, 0, 0, 0])

    def test_source_consumer_combination(self):
        source = native.lightning_indexer_v2(
            self.q, self.k, self.w, 4, candidate_topk_blocks=64
        )
        sparse = native.sparse_lightning_indexer(self.q, self.k, self.w, 4, source[2])
        torch.testing.assert_close(source[0], sparse[0])

    def test_reference_rejects_unsupported_layout(self):
        with self.assertRaises(NotImplementedError):
            native.lightning_indexer_v2(self.q, self.k, self.w, 1, layout_q="TND")

    def test_candidate_validation(self):
        for size in (1, 3, 128):
            with self.assertRaises(ValueError):
                validation.validate_candidates(16, size, width=64)
        for width in (0, 1, 63, 2049):
            with self.assertRaises(ValueError):
                validation.validate_candidates(16, 8, width=width)
        with self.assertRaises(ValueError):
            validation.validate_candidates(2049, 8, width=64)
        with self.assertRaises(ValueError):
            validation.validate_candidates(1, 8, width=64, length=torch.ones(1))


class TestIndexerPackaging(unittest.TestCase):
    def test_public_dispatch_and_bundled_imports(self):
        # Exercise the real registry/wrappers with hardware discovery isolated.
        packages = {}
        for suffix in (
            "",
            ".operators",
            ".operators.attention",
            ".operators.utils",
            ".implementation",
            ".implementation.torch_native",
            ".implementation.torch_native.attention",
            "._third_party",
            "._third_party.sgl_ops_transformer",
        ):
            name = "sgl_kernel_npu" + suffix
            module = types.ModuleType(name)
            module.__path__ = (
                [str(PACKAGE.joinpath(*suffix.strip(".").split(".")))]
                if suffix
                else [str(PACKAGE)]
            )
            packages[name] = module
        bundled = types.ModuleType(
            "sgl_kernel_npu._third_party.sgl_ops_transformer.ops"
        )
        for name in (
            "lightning_indexer",
            "sparse_lightning_indexer",
            "lightning_indexer_metadata",
        ):
            setattr(bundled, name, Mock(return_value=name))
        packages[bundled.__name__] = bundled
        with patch.dict(sys.modules, packages):
            generation = importlib.import_module(
                "sgl_kernel_npu.operators.utils.generation"
            )
            cann = importlib.import_module("sgl_kernel_npu.operators.utils.cann")
            hardware = generation.Hardware(2201, generation.Generation.ARCH22)
            with patch.object(
                generation, "get_hardware", return_value=hardware
            ), patch.object(cann, "detect_cann_version", return_value="9.2.0"):
                dense = importlib.import_module(
                    "sgl_kernel_npu.operators.attention.lightning_indexer_v2"
                ).lightning_indexer_v2
                sparse = importlib.import_module(
                    "sgl_kernel_npu.operators.attention.sparse_lightning_indexer"
                ).sparse_lightning_indexer
                metadata = importlib.import_module(
                    "sgl_kernel_npu.operators.attention.lightning_indexer_v2_metadata"
                ).lightning_indexer_v2_metadata
                q = torch.ones(1, 1, 1, 128, dtype=torch.float16)
                k = torch.ones(1, 8, 1, 128, dtype=torch.float16)
                w = torch.ones(1, 1, 1, dtype=torch.float32)
                self.assertEqual(dense(q, k, w, 1), "lightning_indexer")
                self.assertEqual(len(dense(q, k, w, 1, impl="torch_native")), 2)
                candidates = torch.full((1, 1, 1, 64), -1, dtype=torch.int32)
                self.assertEqual(
                    sparse(q, k, w, 1, candidates), "sparse_lightning_indexer"
                )
                self.assertEqual(metadata(1, 1, 128, 1), "lightning_indexer_metadata")
                self.assertIs(
                    bundled.sparse_lightning_indexer.call_args.kwargs[
                        "candidate_block_indices"
                    ],
                    candidates,
                )
                with self.assertRaises(ValueError):
                    dense(q, k, w, 1, candidate_topk_blocks=63)
                with self.assertRaises(TypeError):
                    dense(q.float(), k, w, 1)

    def test_profile_and_staging(self):
        profile = json.loads(
            (ROOT / "third_party/profiles/cann-9.2.0/ops_transformer.json").read_text()
        )
        self.assertEqual(profile["commit"], "e7da18105f08b06e01ef0fff78f439098d483746")
        self.assertEqual(profile["ops"]["arch35"], ["flash_mla", "flash_mla_metadata"])
        sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
        try:
            from build_tools.third_party import get_adapter, resolve_plan

            adapter = get_adapter("ops_transformer")
            with patch.object(
                adapter, "detect_soc_version", return_value="ascend910_93"
            ):
                plan = resolve_plan("9.2.0", "2201", third_party=("ops_transformer",))
            item = next(p for p in plan if p["name"] == "sgl_ops_transformer")
            self.assertEqual(
                set(item["ops"]),
                {
                    "lightning_indexer_v2",
                    "lightning_indexer_v2_metadata",
                    "sparse_lightning_indexer",
                },
            )
            self.assertTrue(item["has_aicpu"])
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                wheel = base / "bindings.whl"
                prefix = "cann_ops_transformer_" + item["vendor_name"]
                with zipfile.ZipFile(wheel, "w") as archive:
                    for op in (
                        "lightning_indexer_v2",
                        "sparse_lightning_indexer",
                        "unselected",
                    ):
                        archive.writestr(
                            prefix + "/ops/attention/" + op + "/__init__.py",
                            "from cann_ops_transformer.op_builder import OpBuilder\n",
                        )
                    archive.writestr(prefix + "/op_builder/__init__.py", "# builder\n")
                adapter.stage_python(wheel, item, base)
                staged = base / "_third_party/sgl_ops_transformer"
                imports = (staged / "ops/__init__.py").read_text()
                self.assertIn("attention.lightning_indexer_v2", imports)
                self.assertIn("attention.sparse_lightning_indexer", imports)
                self.assertNotIn("unselected", imports)
                source = (
                    staged / "ops/attention/lightning_indexer_v2/__init__.py"
                ).read_text()
                self.assertIn(
                    "sgl_kernel_npu._third_party.sgl_ops_transformer.op_builder", source
                )
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
