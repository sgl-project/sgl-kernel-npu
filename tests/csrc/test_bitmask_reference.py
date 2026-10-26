"""CPU semantic and dispatch tests; these do not emulate an AscendC Kernel."""
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu"


def load_reference():
    spec = importlib.util.spec_from_file_location("bitmask_reference", PACKAGE / "implementation/torch_native/sampler/apply_token_bitmask.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_token_bitmask


@unittest.skipIf(torch is None, "CPU PyTorch required")
class NativeSemanticsTests(unittest.TestCase):
    def test_signed_words_missing_width_and_row_selection(self):
        native = load_reference()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for indices in (None, torch.tensor([], dtype=torch.int32), torch.tensor([0, 2]), torch.tensor([-1, 0, 0])):
                values = torch.arange(3 * 65).reshape(3, 65).to(dtype)
                mask = torch.tensor([[0, -1], [-2147483648, 1], [0x55555555, 0]], dtype=torch.int32)
                expected = values.clone()
                rows = range(3) if indices is None or indices.numel() == 0 else indices.tolist()
                for row in rows:
                    for column in range(65):
                        word = int(mask[row, column // 32]) if column // 32 < 2 else 0
                        if not ((word >> (column % 32)) & 1):
                            expected[row, column] = float("-inf")
                original_mask = mask.clone()
                output = native(values, mask, indices)
                self.assertEqual(output.data_ptr(), values.data_ptr())
                self.assertTrue(torch.equal(output, expected))
                self.assertTrue(torch.equal(mask, original_mask))

    def test_zero_rows_and_extra_words(self):
        native = load_reference()
        values = torch.empty(0, 320)
        self.assertIs(native(values, torch.empty(0, 10, dtype=torch.int32)), values)
        values = torch.ones(2, 31)
        self.assertTrue(torch.equal(native(values, torch.full((2, 4), -1, dtype=torch.int32)), torch.ones(2, 31)))

    def test_invalid_dtype_is_not_silently_converted(self):
        native = load_reference()
        with self.assertRaisesRegex(RuntimeError, "int32"):
            native(torch.ones(1, 32), torch.ones(1, 1, dtype=torch.int64))


@unittest.skipIf(torch is None, "CPU PyTorch required")
class PublicDispatchTests(unittest.TestCase):
    def setUp(self):
        # Import the real registry and implementations without package initialization.
        # Fake hardware answers exercise dispatch only; no NPU execution is claimed.
        package = ModuleType("sgl_kernel_npu")
        package.__path__ = [str(PACKAGE)]
        self.modules = patch.dict(sys.modules, {"sgl_kernel_npu": package,
            "acl": SimpleNamespace(rt=SimpleNamespace(get_device_info=lambda *_: (2201, 0)))})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.operator = importlib.import_module("sgl_kernel_npu.operators.sampler").apply_token_bitmask
        self.adapter = importlib.import_module("sgl_kernel_npu.implementation.ascendc.sampler.apply_token_bitmask")
        self.adapter._operator.cache_clear()
        self.addCleanup(self.adapter._operator.cache_clear)

    def test_auto_and_ascendc_share_one_entry(self):
        registry = importlib.import_module("sgl_kernel_npu.operators.utils.registry")
        self.assertIs(registry.get_impl(self.operator, "auto"), registry.get_impl(self.operator, "ascendc"))
        sentinel = object()
        with patch.object(torch.ops, "npu", SimpleNamespace(apply_token_bitmask=lambda *args: sentinel)):
            self.assertIs(self.operator(None, None), sentinel)
        with self.assertRaises(registry.ImplementationNotFoundError):
            self.operator(None, None, impl="missing")

    def test_missing_native_fails_without_fallback(self):
        with patch.object(torch.ops, "npu", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "does not register"):
                self.operator(None, None)
        values = torch.ones(1, 32)
        self.assertIs(self.operator(values, torch.zeros(1, 1, dtype=torch.int32), impl="torch_native"), values)
        self.assertTrue(torch.isneginf(values).all())

    def test_subset_manifest_is_checked_before_torch_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "_build_manifest.json").write_text('{"operators": [], "arch_version": "2201"}')
            fake_source = root / "implementation/ascendc/sampler/apply_token_bitmask.py"
            with patch.object(self.adapter, "__file__", str(fake_source)):
                with self.assertRaisesRegex(RuntimeError, "does not include"):
                    self.operator(None, None)


@unittest.skipIf(torch is None, "CPU PyTorch required")
class FreshInputTests(unittest.TestCase):
    def test_accuracy_and_profile_inputs_are_independent(self):
        sys.path.insert(0, str(ROOT))
        self.addCleanup(lambda: sys.path.remove(str(ROOT)))
        from tests.python.utils.accuracy import check_accuracy
        calls = []

        def factory():
            calls.append("prepare")
            return (torch.zeros(2),), {}

        def operator(values, *, impl):
            values.add_(1)
            return values

        with patch("tests.python.utils.accuracy.require_distinct_impls"):
            executor = check_accuracy(operator, impl="dut", impl_ref="ref", input_factory=factory)
        self.assertTrue(executor.accuracy_checked)
        executor.prepare_calls(3)
        prepared = len(calls)
        for _ in range(3):
            self.assertTrue(torch.equal(executor(), torch.ones(2)))
        self.assertEqual(len(calls), prepared, "Input preparation occurred during profiled calls")
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            executor()


if __name__ == "__main__":
    unittest.main()
