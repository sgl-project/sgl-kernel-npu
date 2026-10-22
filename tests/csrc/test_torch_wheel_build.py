"""Portable packaging checks; synthetic libraries never emulate NPU execution."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

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
        self.python = self.root / "python/sgl_kernel_npu"
        shutil.copytree(ROOT / "python/sgl_kernel_npu/build_tools", self.python / "build_tools",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "python/sgl_kernel_npu/setup.py", self.python)
        source = self.python / "sgl_kernel_npu"
        source.mkdir()
        for name in ("__init__.py", "_cann_runtime.py"):
            shutil.copy2(ROOT / "python/sgl_kernel_npu/sgl_kernel_npu" / name, source / name)
        directory = self.root / "build/cann"
        self.vendor = directory / "installed/vendors/sgl_kernel_npu"
        for name in ("op_api/lib/libcust_opapi.so", "op_api/include/aclnn_fixture.h",
                     "op_impl/ai_core/tbe/kernel/ascend910_93/fixture.o",
                     "op_impl/ai_core/tbe/kernel/config/ascend910_93/fixture.json"):
            path = self.vendor / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic NPU artifact\n")
        self.cann = cann.CannBuild(directory, ("apply_token_bitmask",), vendor=self.vendor)
        self.package = self.root / "build/torch/package/sgl_kernel_npu"

    def staged(self):
        torch_wheel.stage_runtime(self.vendor, self.package, self.root)
        (self.package / "lib").mkdir()
        for name in ("libsgl_kernel_npu.so", "libsgl_kernel_npu_compare.so"):
            (self.package / "lib" / name).write_bytes(b"synthetic Torch library\n")
        (self.package / "_cann/selected-operators.txt").write_text("apply_token_bitmask\n")
        (self.package / "_cann/soc.txt").write_text("ascend910_93\n")

    def test_missing_cann_cannot_start_torch(self):
        self.cann.vendor = None
        with self.assertRaisesRegex(RuntimeError, "built and installed"):
            torch_wheel.build_torch(self.cann, "ascend910_93", source_root=self.root)

    def test_runtime_preserves_other_vendor_paths(self):
        self.staged()
        prepare = runpy.run_path(str(self.package / "_cann_runtime.py"))["prepare"]
        with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "/other/vendor"}):
            vendor = prepare()
            prepare()
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], str(vendor) + ":/other/vendor")
        self.assertTrue((vendor / "op_api/lib/libcust_opapi.so").is_file())

    def test_runtime_rejects_missing_embedded_library(self):
        self.staged()
        (self.package / "_cann/vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so").unlink()
        prepare = runpy.run_path(str(self.package / "_cann_runtime.py"))["prepare"]
        with self.assertRaisesRegex(RuntimeError, "missing its CANN library"):
            prepare()

    def test_real_cmake_handoff_with_synthetic_extension(self):
        project = self.root / "csrc/aclnn_torch_adapter"
        project.mkdir(parents=True)
        (project / "fixture.cpp").write_text("int fixture() { return 0; }\n")
        (project / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(synthetic_torch_handoff LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release" OR
   NOT SGL_CANN_SOC STREQUAL "ascend910_93" OR
   NOT SGL_CANN_OPERATORS STREQUAL "apply_token_bitmask" OR
   NOT EXISTS "${SGL_CANN_VENDOR}/op_api/lib/libcust_opapi.so")
  message(FATAL_ERROR "Coordinator lost the CANN build selection or vendor")
endif()
foreach(target IN ITEMS sgl_kernel_npu sgl_kernel_npu_compare)
  add_library(${target} SHARED fixture.cpp)
  set_target_properties(${target} PROPERTIES SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY "${SGL_TORCH_PACKAGE}/lib")
endforeach()
''')
        with contextlib.redirect_stdout(io.StringIO()):
            result = torch_wheel.build_torch(self.cann, "ascend910_93", jobs=2, source_root=self.root)
        self.assertEqual(result, self.package)
        self.assertTrue((result / "lib/libsgl_kernel_npu_compare.so").is_file())
        self.assertEqual((result / "_cann/selected-operators.txt").read_text(), "apply_token_bitmask\n")

    def test_torch_build_failure_does_not_report_a_package(self):
        with patch.object(torch_wheel, "run", side_effect=RuntimeError("compile failed")):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                torch_wheel.build_torch(self.cann, "ascend910_93", source_root=self.root)

    def test_real_setuptools_wheel_and_tamper_detection(self):
        self.staged()
        # Only version metadata is needed for packaging; no fake NPU execution.
        stubs = self.root / "metadata_stubs"
        stubs.mkdir()
        for name, version in (("torch", "2.10.0+cpu"), ("torch_npu", "2.10.0.post6")):
            (stubs / (name + ".py")).write_text("__version__ = " + repr(version) + "\n")
        with patch.dict(os.environ, {"PYTHONPATH": str(stubs), "PYTHONPYCACHEPREFIX": str(self.root / "pycache")}):
            with contextlib.redirect_stdout(io.StringIO()):
                wheel = torch_wheel.build_wheel(self.package, source_root=self.root)
        staged = self.root / "build/wheel/package/sgl_kernel_npu"
        torch_wheel.verify_wheel(wheel, staged)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            self.assertFalse(any("compare" in name for name in names))
            self.assertFalse(any(name.endswith(".run") for name in names))
            metadata = archive.read(next(name for name in names if name.endswith(".dist-info/METADATA")))
            self.assertIn(b"torch", metadata)
            self.assertIn(b"2.10.0.post6", metadata)
        self.assertFalse(list(self.python.glob("*.egg-info")), "Packaging polluted the legacy source directory")
        (staged / "lib/libsgl_kernel_npu.so").write_bytes(b"wrong build")
        with self.assertRaisesRegex(RuntimeError, "differs from this build"):
            torch_wheel.verify_wheel(wheel, staged)

    def test_declarations_cannot_claim_a_wheel(self):
        result = subprocess.run([sys.executable, str(ROOT / "python/sgl_kernel_npu/build_tools/cann.py"),
                                 "--soc", "ascend910_93", "--declarations-only", "--artifact", "wheel"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot build Torch", result.stderr)


if __name__ == "__main__":
    unittest.main()
