"""CPU mathematical, mutation and experimental-profile checks."""

import importlib
import importlib.util
import math
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
spec = importlib.util.spec_from_file_location(
    "compressor_reference",
    PACKAGE / "implementation/torch_native/attention/compressor_v2.py",
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class TestCompressorReference(unittest.TestCase):
    def inputs(self, values, *, blocks=1, capacity=4):
        x = torch.tensor(values, dtype=torch.float16).reshape(1, -1, 1)
        return (
            x,
            torch.ones(1, 1, dtype=torch.float16),
            torch.zeros(1, 1, dtype=torch.float16),
            torch.full((blocks, capacity, 2), -19.0),
        )

    def test_uniform_gate_mean_and_physical_zero(self):
        x, kv, gate, cache = self.inputs([1, 3, 5, 7])
        result = native.compressor_v2(
            x, kv, gate, cache, 4, state_block_table=torch.tensor([0])
        )
        self.assertEqual(result.item(), 4)
        torch.testing.assert_close(cache[0, :, 0], torch.tensor([1.0, 3.0, 5.0, 7.0]))
        self.assertTrue((cache[0, :, 1] == 0).all())

    def test_softmax_is_per_feature_along_tokens(self):
        x, kv, gate, cache = self.inputs([1, 2])
        gate.fill_(1)
        result = native.compressor_v2(
            x, kv, gate, cache, 2, state_block_table=torch.tensor([0])
        )
        self.assertAlmostEqual(result.item(), 1 + 1 / (1 + math.exp(-1)), delta=0.001)

    def test_history_and_untouched_block(self):
        x, kv, gate, cache = self.inputs([10, 14], blocks=2)
        cache[1, :2, 0] = torch.tensor([2.0, 6.0])
        cache[1, :2, 1] = 0
        before = cache[0].clone()
        result = native.compressor_v2(
            x,
            kv,
            gate,
            cache,
            4,
            state_block_table=torch.tensor([1]),
            start_pos=torch.tensor([2]),
        )
        self.assertEqual(result.item(), 8)
        torch.testing.assert_close(cache[0], before)
        torch.testing.assert_close(cache[1, :, 0], torch.tensor([2.0, 6.0, 10.0, 14.0]))

    def test_two_step_continuation(self):
        x, kv, gate, cache = self.inputs([1, 3])
        native.compressor_v2(x, kv, gate, cache, 4, state_block_table=torch.tensor([0]))
        x.fill_(0)
        x[0, :, 0] = torch.tensor([5.0, 7.0])
        result = native.compressor_v2(
            x,
            kv,
            gate,
            cache,
            4,
            state_block_table=torch.tensor([0]),
            start_pos=torch.tensor([2]),
        )
        self.assertEqual(result.item(), 4)

    def test_wrap_writes_only_final_capacity(self):
        x, kv, gate, cache = self.inputs([1, 2, 3, 4, 5, 6])
        result = native.compressor_v2(
            x, kv, gate, cache, 4, state_block_table=torch.tensor([0])
        )
        self.assertEqual(result[0, 0, 0].item(), 2.5)
        torch.testing.assert_close(cache[0, :, 0], torch.tensor([5.0, 6.0, 3.0, 4.0]))

    def test_packed_outputs_are_compacted(self):
        x, kv, gate, cache = self.inputs(range(1, 9), blocks=2)
        result = native.compressor_v2(
            x.reshape(-1, 1),
            kv,
            gate,
            cache,
            2,
            state_block_table=torch.tensor([0, 1]),
            cu_seqlens=torch.tensor([0, 3, 8]),
        )
        self.assertEqual(result.shape, (6, 1))
        self.assertEqual(result[:3, 0].tolist(), [1.5, 4.5, 6.5])

    def test_seqused_and_empty_preserve_unused_state(self):
        x, kv, gate, cache = self.inputs([1, 2, 3, 4])
        before = cache.clone()
        native.compressor_v2(
            x,
            kv,
            gate,
            cache,
            4,
            state_block_table=torch.tensor([0]),
            seqused=torch.tensor([0]),
        )
        torch.testing.assert_close(cache, before)
        result = native.compressor_v2(
            x[:, :0], kv, gate, cache, 4, state_block_table=torch.tensor([0])
        )
        self.assertEqual(result.numel(), 0)
        torch.testing.assert_close(cache, before)

    def test_duplicate_blocks_are_invalid(self):
        x, kv, gate, cache = self.inputs([1, 2], blocks=2)
        with self.assertRaises(ValueError):
            native.compressor_v2(
                x.expand(2, -1, -1),
                kv,
                gate,
                cache,
                4,
                state_block_table=torch.tensor([0, 0]),
            )


class TestExperimentalProfile(unittest.TestCase):
    def test_public_validation_and_ds41_binding(self):
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
            "._third_party.sgl_ops_transformer_experimental",
            "._third_party.sgl_ops_transformer_experimental.ops",
        ):
            name = "sgl_kernel_npu" + suffix
            module = types.ModuleType(name)
            module.__path__ = (
                [str(PACKAGE.joinpath(*suffix.strip(".").split(".")))]
                if suffix
                else [str(PACKAGE)]
            )
            packages[name] = module
        ds41 = types.ModuleType(
            "sgl_kernel_npu._third_party.sgl_ops_transformer_experimental.ops.ds41"
        )
        ds41.compressor = Mock(return_value="bundled")
        packages[ds41.__name__] = ds41
        with patch.dict(sys.modules, packages):
            generation = importlib.import_module(
                "sgl_kernel_npu.operators.utils.generation"
            )
            cann = importlib.import_module("sgl_kernel_npu.operators.utils.cann")
            with patch.object(
                generation,
                "get_hardware",
                return_value=generation.Hardware(2201, generation.Generation.ARCH22),
            ), patch.object(cann, "detect_cann_version", return_value="9.2.0"):
                public = importlib.import_module(
                    "sgl_kernel_npu.operators.attention.compressor_v2"
                ).compressor_v2
                x = torch.ones(1, 4, 1024, dtype=torch.float16)
                w = torch.zeros(128, 1024, dtype=x.dtype)
                cache = torch.zeros(1, 4, 256)
                blocks = torch.tensor([0], dtype=torch.int32)
                self.assertEqual(
                    public(x, w, w, cache, state_block_table=blocks), "bundled"
                )
                self.assertIs(ds41.compressor.call_args.args[3], cache)
                result = public(
                    x, w, w, cache, state_block_table=blocks, impl="torch_native"
                )
                self.assertEqual(result.shape, (1, 1, 128))
                for kwargs in (
                    {},
                    {"cmp_ratio": 3, "state_block_table": blocks},
                    {"state_block_table": blocks.long()},
                ):
                    with self.assertRaises(ValueError):
                        public(x, w, w, cache, **kwargs)
                with self.assertRaises(TypeError):
                    public(x, w, w, cache.half(), state_block_table=blocks)

    def test_two_profiles_stage_without_collisions(self):
        sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
        try:
            from build_tools.third_party import get_adapter, resolve_plan

            adapter = get_adapter("ops_transformer")
            with patch.object(
                adapter, "detect_soc_version", return_value="ascend910_93"
            ):
                plan = resolve_plan("9.2.0", "2201")
            self.assertEqual(len(plan), 2)
            self.assertEqual(len({item["vendor_name"] for item in plan}), 2)
            experimental = next(p for p in plan if p["name"].endswith("experimental"))
            self.assertEqual(
                experimental["commit"], "e7da18105f08b06e01ef0fff78f439098d483746"
            )
            self.assertFalse(experimental["has_aicpu"])
            self.assertEqual(experimental["ops"], ["compressor_v2"])
            self.assertTrue(
                all("--experimental" in c for c in experimental["commands"])
            )
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                for item in plan:
                    wheel = base / (item["name"] + ".whl")
                    prefix = "cann_ops_transformer_" + item["vendor_name"]
                    with zipfile.ZipFile(wheel, "w") as archive:
                        op = item["ops"][0]
                        archive.writestr(
                            prefix + "/ops/attention/" + op + "/__init__.py",
                            "from cann_ops_transformer.op_builder import OpBuilder\n"
                            'parent = sys.modules["cann_ops_transformer.ops"].__name__\n'
                            'schema = "cann_ops_transformer::ds41.compressor"\n',
                        )
                    adapter.stage_python(wheel, item, base)
                packages = list((base / "_third_party").iterdir())
                self.assertEqual(len(packages), 2)
                imports = (
                    base
                    / "_third_party/sgl_ops_transformer_experimental/ops/attention/compressor_v2/__init__.py"
                ).read_text()
                self.assertIn(
                    "sgl_kernel_npu._third_party.sgl_ops_transformer_experimental.op_builder",
                    imports,
                )
                self.assertIn(
                    'sys.modules["sgl_kernel_npu._third_party.sgl_ops_transformer_experimental.ops"]',
                    imports,
                )
                self.assertIn('"cann_ops_transformer::ds41.compressor"', imports)
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
