"""Portable packaging checks; synthetic libraries never emulate NPU execution."""

import builtins
import contextlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
from build_tools import cann, torch_wheel


class TorchWheelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo with spaces"
        self.root.mkdir()
        shutil.copy2(ROOT / "config.ini", self.root)
        shutil.copy2(ROOT / "setup.py", self.root)
        (self.root / "csrc/cmake").mkdir(parents=True)
        shutil.copy2(ROOT / "csrc/cmake/architectures.json", self.root / "csrc/cmake")
        self.python = self.root / "python/sgl_kernel_npu"
        shutil.copytree(
            ROOT / "python/sgl_kernel_npu/build_tools",
            self.python / "build_tools",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(ROOT / "python/sgl_kernel_npu/setup.py", self.python)
        shutil.copytree(
            ROOT / "python/sgl_kernel_npu/target_providers",
            self.python / "target_providers",
        )
        source = self.python / "sgl_kernel_npu"
        source.mkdir()
        shutil.copy2(
            ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/__init__.py",
            source / "__init__.py",
        )
        (source / "operators/sampler").mkdir(parents=True)
        (source / "operators/sampler/fixture.py").write_text(
            "def example(): return 42\n"
        )
        (source / "implementation/triton").mkdir(parents=True)
        (source / "implementation/triton/fixture.py").write_text(
            "# Python implementation fixture\n"
        )
        directory = self.root / "build/cann"
        self.vendor = directory / "installed/vendors/sgl_kernel_npu"
        for name in (
            "op_api/lib/libcust_opapi.so",
            "op_api/include/aclnn_fixture.h",
            "op_impl/ai_core/tbe/kernel/ascend910_93/fixture.o",
            "op_impl/ai_core/tbe/kernel/config/ascend910_93/fixture.json",
        ):
            path = self.vendor / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic NPU artifact\n")
        self.cann = cann.CannBuild(
            directory, ("apply_token_bitmask",), vendor=self.vendor
        )
        self.package = self.root / "build/torch/package/sgl_kernel_npu"

    def staged(self):
        torch_wheel.stage_runtime(self.vendor, self.package, self.root)
        (self.package / "lib").mkdir()
        for name in ("libsgl_kernel_npu.so",):
            (self.package / "lib" / name).write_bytes(b"synthetic Torch library\n")
        (self.package / "_cann/selected-operators.txt").write_text(
            "apply_token_bitmask\n"
        )
        (self.package / "_cann/arch-version.txt").write_text("2201\n")
        (self.package / "_build_manifest.json").write_text(
            json.dumps(
                {
                    "selection_kind": "custom",
                    "operators": ["apply_token_bitmask"],
                    "arch_version": "2201",
                }
            )
        )

    def test_missing_cann_cannot_start_torch(self):
        self.cann.vendor = None
        with self.assertRaisesRegex(RuntimeError, "built and installed"):
            torch_wheel.build_torch(self.cann, "2201", source_root=self.root)

    def load_runtime(self, events):
        original_import = builtins.__import__
        torch = SimpleNamespace(
            ops=SimpleNamespace(
                load_library=lambda path: events.append(("load_library", path))
            )
        )

        def import_module(name, *args, **kwargs):
            if name in ("torch", "torch_npu"):
                events.append((name, os.environ.get("ASCEND_CUSTOM_OPP_PATH")))
                return torch if name == "torch" else SimpleNamespace()
            return original_import(name, *args, **kwargs)

        # Exercise the real package initializer without loading NPU libraries.
        with patch("builtins.__import__", side_effect=import_module):
            runpy.run_path(str(self.package / "__init__.py"))

    def test_runtime_preserves_other_vendor_paths(self):
        self.staged()
        vendor = (self.package / "_cann/vendors/sgl_kernel_npu").resolve()
        expected = str(vendor) + os.pathsep + "/other/vendor"
        previous = os.pathsep.join(("", "/other/vendor", str(vendor), str(vendor), ""))
        with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": previous}):
            for _ in range(2):
                events = []
                self.load_runtime(events)
                self.assertEqual(
                    events,
                    [
                        ("torch", expected),
                        ("torch_npu", expected),
                        (
                            "load_library",
                            str(self.package / "lib/libsgl_kernel_npu.so"),
                        ),
                    ],
                )
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], expected)
        self.assertTrue((vendor / "op_api/lib/libcust_opapi.so").is_file())

    def test_runtime_rejects_missing_embedded_library(self):
        self.staged()
        (
            self.package / "_cann/vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so"
        ).unlink()
        events = []
        with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "/other/vendor"}):
            with self.assertRaisesRegex(RuntimeError, "missing its CANN library"):
                self.load_runtime(events)
            self.assertEqual(
                events, [], "Missing CANN library must fail before importing Torch"
            )
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], "/other/vendor")

    def test_runtime_without_cann_keeps_legacy_loading(self):
        self.staged()
        shutil.rmtree(self.package / "_cann")
        events = []
        with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "/other/vendor"}):
            self.load_runtime(events)
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], "/other/vendor")
        self.assertEqual(
            events,
            [
                ("torch", "/other/vendor"),
                ("torch_npu", "/other/vendor"),
                ("load_library", str(self.package / "lib/libsgl_kernel_npu.so")),
            ],
        )

    def test_real_cmake_handoff_with_synthetic_extension(self):
        project = self.root / "csrc/aclnn_torch_adapter"
        project.mkdir(parents=True)
        (project / "fixture.cpp").write_text("int fixture() { return 0; }\n")
        (project / "CMakeLists.txt").write_text(
            """
cmake_minimum_required(VERSION 3.20)
project(synthetic_torch_handoff LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release" OR
   NOT SGL_ARCH_VERSION STREQUAL "2201" OR
   NOT SGL_CANN_OPERATORS STREQUAL "apply_token_bitmask" OR
   NOT EXISTS "${SGL_CANN_VENDOR}/op_api/lib/libcust_opapi.so")
  message(FATAL_ERROR "Coordinator lost the CANN build selection or vendor")
endif()
foreach(target IN ITEMS sgl_kernel_npu)
  add_library(${target} SHARED fixture.cpp)
  set_target_properties(${target} PROPERTIES SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY "${SGL_TORCH_LIBRARY_DIR}")
endforeach()
"""
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = torch_wheel.build_torch(
                self.cann, "2201", jobs=2, source_root=self.root
            )
        self.assertEqual(result, self.package)
        self.assertFalse((result / "lib/libsgl_kernel_npu_compare.so").exists())
        self.assertEqual(
            (result / "_cann/selected-operators.txt").read_text(),
            "apply_token_bitmask\n",
        )
        library = self.root / "build/torch/lib/libsgl_kernel_npu.so"
        timestamp = library.stat().st_mtime_ns
        with contextlib.redirect_stdout(io.StringIO()):
            torch_wheel.build_torch(self.cann, "2201", jobs=2, source_root=self.root)
        self.assertEqual(
            library.stat().st_mtime_ns,
            timestamp,
            "Unchanged Torch build relinked the library",
        )
        with patch.object(
            torch_wheel.importlib.metadata, "version", return_value="different-version"
        ):
            with self.assertRaisesRegex(RuntimeError, "environment changed"):
                torch_wheel.build_torch(self.cann, "2201", source_root=self.root)
        self.assertEqual(library.stat().st_mtime_ns, timestamp)

    def test_torch_build_failure_does_not_report_a_package(self):
        with patch.object(
            torch_wheel, "run", side_effect=RuntimeError("compile failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                torch_wheel.build_torch(self.cann, "2201", source_root=self.root)

    def test_real_setuptools_wheel_and_tamper_detection(self):
        self.staged()
        # Only version metadata is needed for packaging; no fake NPU execution.
        stubs = self.root / "metadata_stubs"
        stubs.mkdir()
        for name, version in (("torch", "2.10.0+cpu"), ("torch_npu", "2.10.0.post6")):
            (stubs / (name + ".py")).write_text("__version__ = " + repr(version) + "\n")
        with patch.dict(
            os.environ,
            {
                "PYTHONPATH": str(stubs),
                "PYTHONPYCACHEPREFIX": str(self.root / "pycache"),
            },
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                wheel = torch_wheel.build_wheel(self.package, source_root=self.root)
        staged = self.root / "build/wheel/package/sgl_kernel_npu"
        torch_wheel.verify_wheel(wheel, staged)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            self.assertFalse(any("compare" in name for name in names))
            self.assertIn("sgl_kernel_npu/operators/sampler/fixture.py", names)
            self.assertIn("sgl_kernel_npu/implementation/triton/fixture.py", names)
            self.assertIn("sgl_kernel_npu/norm/gemma_rmsnorm.py", names)
            self.assertIn("1refactor2201custom", wheel.name)
            self.assertFalse(any(name.endswith(".run") for name in names))
            metadata = archive.read(
                next(name for name in names if name.endswith(".dist-info/METADATA"))
            )
            self.assertIn(b"torch", metadata)
            self.assertIn(b"2.10.0.post6", metadata)
        self.assertFalse(
            list(self.python.glob("*.egg-info")),
            "Packaging polluted the legacy source directory",
        )
        (staged / "lib/libsgl_kernel_npu.so").write_bytes(b"wrong build")
        with self.assertRaisesRegex(RuntimeError, "differs from this build"):
            torch_wheel.verify_wheel(wheel, staged)

    def test_custom_build_tag_is_order_independent(self):
        manifest = {
            "selection_kind": "custom",
            "operators": ["b", "a"],
            "arch_version": "2201",
        }
        tag = torch_wheel.wheel_build_tag(manifest)
        manifest["operators"].reverse()
        self.assertEqual(tag, torch_wheel.wheel_build_tag(manifest))
        manifest["selection_kind"] = "all"
        self.assertEqual(torch_wheel.wheel_build_tag(manifest), "1refactor2201all")

    def test_restage_discards_stale_python_and_native_files(self):
        self.staged()
        stale = self.package / "operators/removed.py"
        stale.write_text("# removed operator")
        torch_wheel.stage_runtime(self.vendor, self.package, self.root)
        self.assertFalse(stale.exists())
        self.assertFalse((self.package / "lib").exists())

    def test_declarations_only_is_not_an_artifact(self):
        from build_tools.refactor import arguments

        args = arguments(["--arch-version", "2201", "--declarations-only"])
        self.assertTrue(args.declarations_only)


if __name__ == "__main__":
    unittest.main()
