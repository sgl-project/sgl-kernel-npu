"""Host-only tests; neither CANN nor Torch is imported."""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/sgl_kernel_npu"))

from build_tools.third_party import (
    extract_python,
    get_adapter,
    resolve_plan,
    stage_dependencies,
)
from build_tools.torch_wheel import wheel_build_tag

inspect_vendor = get_adapter("ops_transformer").inspect_vendor


class ThirdPartyBuildTests(unittest.TestCase):
    def test_adapter_selection(self):
        self.assertEqual(resolve_plan("9.2.0", "3510")[0]["builder"], "ops_transformer")
        with self.assertRaisesRegex(ValueError, "Unsupported third-party builder"):
            get_adapter("../outside")
        with self.assertRaisesRegex(ValueError, "Unsupported third-party builder"):
            get_adapter("cann_recipes_infer")

    def test_vendor_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor = Path(tmp)
            item = resolve_plan("9.2.0", "3510")[0]
            resources = {
                "op_api/lib/libcust_opapi.so": "library",
                "op_impl/cpu/config/cust_aicpu_kernel.json": json.dumps(
                    {"FlashMlaMetadata": {"opInfo": {"kernelSo": "libmetadata.so"}}}
                ),
                "op_impl/cpu/aicpu_kernel/impl/libmetadata.so": "library",
                "op_impl/ai_core/tbe/kernel/config/ascend950/flash_mla.json": "{}",
                "op_impl/ai_core/tbe/kernel/ascend950/flash_mla/kernel.o": "binary",
                "op_impl/ai_core/tbe/kernel/ascend950/flash_mla/kernel.json": "{}",
            }
            for relative, content in resources.items():
                path = vendor / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            inspect_vendor(vendor, item)
            with self.assertRaisesRegex(RuntimeError, "has_aicpu"):
                inspect_vendor(vendor, dict(item, has_aicpu=False))
            with self.assertRaisesRegex(RuntimeError, "No installed"):
                inspect_vendor(vendor, dict(item, ops=["missing_op"]))
            for relative in resources:
                with self.subTest(missing=relative):
                    path = vendor / relative
                    path.unlink()
                    with self.assertRaises(RuntimeError):
                        inspect_vendor(vendor, item)
                    path.write_text(resources[relative])

    def test_plan(self):
        item = resolve_plan("9.2.0", "3510", jobs=3)[0]
        self.assertEqual(item["name"], "slg_ops_transformer")
        self.assertIn("--ops=flash_mla,flash_mla_metadata", item["commands"][0])
        self.assertIn("-j3", item["commands"][0])
        self.assertNotIn("--opkernel_aicpu", item["commands"][0])
        self.assertEqual(item["vendor_directory"], item["name"] + "_transformer")
        self.assertEqual(item["python_package"], "cann_ops_transformer_" + item["name"])
        self.assertIn("--ops=flash_mla,flash_mla_metadata", item["commands"][1])
        self.assertNotIn("python_modules", item)

    def test_multiple_python_selections(self):
        source = Path(__file__).resolve().parents[2]
        config = json.loads(
            (
                source / "third_party/profiles/cann-9.2.0/arch35/ops_transformer.json"
            ).read_text()
        )
        config["ops"] = ["flash_mla", "example"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "third_party/profiles/cann-9.2.0/arch35"
            profile.mkdir(parents=True)
            (profile / "ops_transformer.json").write_text(json.dumps(config))
            item = resolve_plan("9.2.0", "3510", source_root=root)[0]
            self.assertIn("--ops=flash_mla,example", item["commands"][1])

    def test_missing_profile(self):
        with self.assertRaisesRegex(ValueError, "No third-party profile"):
            resolve_plan("9.1.0", "3510")

    def test_self_contained_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "third_party/profiles/cann-9.2.0/arch35"
            profile.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "No third-party profile"):
                resolve_plan("9.2.0", "3510", source_root=root)
            config = resolve_plan("9.2.0", "3510")[0]
            (profile / "b.json").write_text(json.dumps(config))
            other = dict(config, name="slg_other")
            (profile / "a.json").write_text(json.dumps(other))
            plan = resolve_plan("9.2.0", "3510", source_root=root)
            self.assertEqual(
                [item["name"] for item in plan], ["slg_other", "slg_ops_transformer"]
            )

    def test_invalid_version(self):
        with self.assertRaises(ValueError):
            resolve_plan("../9.2.0", "3510")

    def test_unsafe_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel = root / "bad.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("pkg/../../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe wheel"):
                extract_python(wheel, "pkg", root / "out")

    def test_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = resolve_plan("9.2.0", "3510")[0]
            item["ops"].append("example")
            vendor = root / "vendor"
            (vendor / "op_api/lib").mkdir(parents=True)
            (vendor / "op_api/lib/libcust_opapi.so").write_bytes(b"library")
            (vendor / "bin").mkdir()
            (vendor / "bin/set_env.bash").write_text("temporary installation path")
            license_file = root / "LICENSE"
            license_file.write_text("upstream license")
            wheel = root / "upstream.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                prefix = item["python_package"]
                archive.writestr(prefix + "/__init__.py", "# upstream")
                archive.writestr(prefix + "/ops/__init__.py", "# discovery")
                archive.writestr(
                    prefix + "/ops/moe/example/__init__.py", "# another module"
                )
                archive.writestr(
                    prefix + "/ops/attention/flash_mla/__init__.py",
                    "# selected operator",
                )
                archive.writestr(
                    prefix + "/op_builder/builder.py",
                    "from cann_ops_transformer.utils.arg_check import wrap_op_module\n",
                )
            package = root / "sgl_kernel_npu"
            manifests = stage_dependencies(
                [dict(config=item, vendor=vendor, wheel=wheel, license=license_file)],
                package,
            )
            target = package / "_cann/vendors" / item["vendor_directory"]
            self.assertFalse((target / "bin").exists())
            self.assertEqual(
                (target / "op_api/lib/libcust_opapi.so").read_bytes(), b"library"
            )
            binding = package / "_third_party/slg_ops_transformer"
            self.assertIn(
                "sgl_kernel_npu._third_party.slg_ops_transformer.utils",
                (binding / "op_builder/builder.py").read_text(),
            )
            self.assertEqual(
                (binding / "ops/__init__.py").read_text(),
                "from .attention.flash_mla import *  # noqa: F403\n"
                "from .moe.example import *  # noqa: F403\n",
            )
            self.assertEqual(manifests[0]["binding"], "upstream_jit")
            json.dumps(manifests)

    def test_tag_tracks_dependencies(self):
        base = dict(selection_kind="all", operators=[], arch_version="3510")
        original = wheel_build_tag(base)
        base["third_party"] = [{"commit": "a" * 40}]
        first = wheel_build_tag(base)
        base["third_party"][0]["commit"] = "b" * 40
        self.assertNotEqual(original, first)
        self.assertNotEqual(first, wheel_build_tag(base))


if __name__ == "__main__":
    unittest.main()
