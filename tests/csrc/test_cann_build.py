"""Portable orchestration tests. Synthetic files do not emulate CANN compilation."""

import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
from build_tools import cann


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class CannBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "repo with spaces"
        self.root.mkdir()
        (self.root / "csrc").mkdir()
        shutil.copy(ROOT / "csrc/CMakeLists.txt", self.root / "csrc")
        shutil.copytree(ROOT / "csrc/cmake", self.root / "csrc/cmake")
        shutil.copytree(ROOT / "csrc/sampling", self.root / "csrc/sampling")
        shutil.copytree(ROOT / "csrc/aclnn_torch_adapter", self.root / "csrc/aclnn_torch_adapter")
        self.directory = self.root / "build/cann"
        self.fail_target = None
        self.fail_install = False

    def build(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return cann.build_cann("ascend910_93", source_root=self.root,
                                   declarations_only=True, **kwargs)

    def fixture_operator(self):
        # An extra declaration, never compiled as a real operator.
        fixture = self.root / "csrc/sampling/fixture"
        fixture.mkdir()
        for name in ("host.cpp", "kernel.cpp"):
            (fixture / name).write_text("// Synthetic source path fixture.\n")
        (fixture / "CMakeLists.txt").write_text("""
sgl_register_operator(ID fixture_op EXECUTION_UNIT AICORE CALL_MODE ACLNN
    CANN_OP_TYPE FixtureOp SUPPORTED_SOCS ascend910_93
    HOST_SOURCES host.cpp KERNEL_SOURCES kernel.cpp)
""")
        with (self.root / "csrc/sampling/CMakeLists.txt").open("a") as stream:
            stream.write("\nadd_subdirectory(fixture)\n")

    def test_real_declarations_without_sdk_or_torch(self):
        with patch.dict(os.environ, {"ASCEND_HOME_PATH": ""}):
            result = self.build()
        self.assertEqual(result.operators, ("apply_token_bitmask",))
        self.assertIsNone(result.package)
        cache = (self.directory / "CMakeCache.txt").read_text()
        self.assertNotIn("CMAKE_CXX_COMPILER:", cache)

    def test_same_selection_keeps_build_directory(self):
        self.build()
        marker = self.directory / "keep"
        marker.touch()
        self.build()
        self.assertTrue(marker.exists())

    def test_switch_clears_readonly_installed_directories(self):
        self.build()
        installed = self.directory / "installed/vendors/fixture"
        installed.mkdir(parents=True)
        (installed / "library.so").write_text("synthetic library")
        installed.chmod(0o555)
        self.build(operators=("apply_token_bitmask",))
        self.assertFalse(installed.exists())

    def test_all_to_subset_clears_old_artifacts(self):
        self.fixture_operator()
        result = self.build()
        self.assertEqual(result.operators, ("apply_token_bitmask", "fixture_op"))
        marker = self.directory / "stale-kernel.o"
        marker.touch()
        result = self.build(operators=("fixture_op",))
        self.assertEqual(result.operators, ("fixture_op",))
        self.assertFalse(marker.exists())

    def test_all_expansion_change_cleans_and_reconfigures(self):
        self.build()
        marker = self.directory / "old-package.run"
        marker.touch()
        self.fixture_operator()
        result = self.build()
        self.assertIn("fixture_op", result.operators)
        self.assertFalse(marker.exists())

    def test_explicit_unknown_id_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Command failed"):
            self.build(operators=("missing_op",))
        self.assertIn("Unknown operator ID", (self.directory / "configure.log").read_text())

    def test_all_unknown_hardware_fails(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            cann.build_cann("fixture_soc_z", source_root=self.root, declarations_only=True)
        self.assertIn("No CANN operators support", (self.directory / "configure.log").read_text())

    def test_switch_from_legacy_cache(self):
        build = self.root / "build"
        build.mkdir()
        (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(self.root) + "\n")
        (build / "old-library.so").touch()
        self.build()
        self.assertFalse((build / "old-library.so").exists())
        self.assertTrue((self.directory / "selected-operators.txt").exists())

    def test_unrelated_build_cache_is_not_deleted(self):
        build = self.root / "build"
        build.mkdir()
        (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/another/project\n")
        with self.assertRaisesRegex(RuntimeError, "Unexpected CMake project"):
            self.build()
        self.assertTrue((build / "CMakeCache.txt").exists())

    def test_symlinked_build_root_is_not_deleted(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "build").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlinked build root"):
            self.build()
        self.assertTrue(outside.exists())

    def fake_run(self, command, log_path, **kwargs):
        # Only verify orchestration order and arguments; no SDK or ELF behavior.
        args = [str(item) for item in command]
        self.calls.append((args, kwargs))
        log_path.write_text("synthetic command log\n")
        if "-S" in args:
            self.directory.joinpath("selected-operators.txt").write_text("apply_token_bitmask\n")
        elif "--build" in args:
            target = args[args.index("--target") + 1]
            if target == self.fail_target:
                raise RuntimeError("synthetic {} failure".format(target))
            if target == "sgl_cann_kernel_check":
                self.kernel_built = True
            elif target == "package":
                self.assertTrue(self.kernel_built, "Kernel compilation must finish before packaging")
                package = self.directory / "package-stage/fixture.run"
                package.parent.mkdir()
                package.write_text("synthetic installer")
            else:
                self.fail("Unexpected build target: " + target)
        elif args[0] == "bash" and self.fail_install:
            raise RuntimeError("synthetic install failure")

    def full_build(self, **kwargs):
        self.calls = []
        self.kernel_built = False
        sdk = self.root / "sdk"
        sdk.mkdir()
        vendor = self.directory / "installed/vendors/sgl_kernel_npu"
        with patch.object(cann.platform, "system", return_value="Linux"), \
                patch.dict(os.environ, {"ASCEND_HOME_PATH": str(sdk)}), \
                patch.object(cann, "run", side_effect=self.fake_run), \
                patch.object(cann, "inspect_installation", return_value=(vendor, [], [])) as inspect, \
                contextlib.redirect_stdout(io.StringIO()):
            result = cann.build_cann("ascend910_93", source_root=self.root, **kwargs)
            inspect.assert_called_once_with(self.directory / "installed", self.directory, "ascend910_93")
        return result

    def test_kernel_then_package_then_private_install(self):
        result = self.full_build()
        self.assertEqual(len(self.calls), 4)
        self.assertIn("sgl_cann_kernel_check", self.calls[1][0])
        self.assertIn("package", self.calls[2][0])
        self.assertIn("--install-path=" + str(self.directory / "installed"), self.calls[3][0])
        self.assertEqual(result.vendor, self.directory / "installed/vendors/sgl_kernel_npu")

    def test_kernel_failure_does_not_package_or_install(self):
        self.fail_target = "sgl_cann_kernel_check"
        with self.assertRaisesRegex(RuntimeError, "synthetic sgl_cann_kernel_check failure"):
            self.full_build()
        self.assertEqual(len(self.calls), 2)

    def test_package_failure_does_not_install(self):
        self.fail_target = "package"
        with self.assertRaisesRegex(RuntimeError, "synthetic package failure"):
            self.full_build()
        self.assertEqual(len(self.calls), 3)

    def test_jobs_applies_to_both_build_steps(self):
        self.full_build(jobs=3)
        for args, _ in self.calls[1:3]:
            self.assertEqual(args[-2:], ["--parallel", "3"])

    def test_install_failure_does_not_report_success(self):
        self.fail_install = True
        with self.assertRaisesRegex(RuntimeError, "synthetic install failure"):
            self.full_build()

    def test_real_cmake_package_requires_explicit_kernel_build(self):
        # Exercise actual CMake commands with synthetic artifacts, not CANN code.
        (self.root / "csrc/CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.20)
project(orchestration_fixture LANGUAGES NONE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/selected-operators.txt" "apply_token_bitmask\\n")
add_custom_target(sgl_cann_kernel_check
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/kernel.marker")
add_custom_target(package
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_SOURCE_DIR}/package.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
''')
        (self.root / "csrc/package.cmake").write_text('''
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/kernel.marker")
    message(FATAL_ERROR "Kernel artifact missing before packaging")
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/package-stage")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/package-stage/fixture.run" "exit 0\\n")
''')
        sdk = self.root / "sdk"
        sdk.mkdir()
        vendor = self.directory / "installed/vendors/sgl_kernel_npu"
        with patch.object(cann.platform, "system", return_value="Linux"), \
                patch.dict(os.environ, {"ASCEND_HOME_PATH": str(sdk)}), \
                patch.object(cann, "inspect_installation", return_value=(vendor, [], [])), \
                contextlib.redirect_stdout(io.StringIO()):
            result = cann.build_cann("ascend910_93", source_root=self.root)
        self.assertTrue((self.directory / "kernel.marker").is_file())
        self.assertTrue(result.package.is_file())
        self.assertTrue((self.directory / "install.log").is_file())

    def test_cli_arguments(self):
        self.assertEqual(cann.parse_operators("all"), ("ALL",))
        self.assertEqual(cann.parse_operators("op_a,op_b"), ("op_a", "op_b"))
        for value in ("", "op_a,", "all,op_a", "op_a,op_a", "op-a", "op_a;op_b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cann.parse_operators(value)


class ShellDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copy(ROOT / "build.sh", self.root)

    def legacy_trace(self, mode):
        names = [
            "parse_arguments", "configure_build_target", "configure_soc_version",
            "setup_ascend_environment", "prepare_deepep_build", "build_cmake_modules",
            "build_deepep_kernels", "build_attentions_kernels", "ensure_wheel_package",
            "make_deepep_package", "make_sgl_kernel_npu_package", "make_attentions_package",
            "build_memory_saver_package",
        ]
        script = "source ./build.sh\n"
        for name in names:
            script += name + '() { echo CALL:' + name + '; }\n'
        script += "BUILD_DEEPEP_MODULE=ON\nBUILD_KERNELS_MODULE=ON\n"
        script += "BUILD_ATTENTIONS_MODULE=ON\nBUILD_MEMORY_SAVER_MODULE=ON\nmain\n"
        env = os.environ.copy()
        env.pop("SGL_BUILD_MODE", None)
        if mode is not None:
            env["SGL_BUILD_MODE"] = mode
        result = subprocess.run(["bash", "-c", script], cwd=self.root, env=env,
                                capture_output=True, text=True)
        return result, names

    def test_default_and_explicit_legacy_keep_all_steps(self):
        for mode in (None, "LEGACY"):
            with self.subTest(mode=mode):
                result, names = self.legacy_trace(mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([line[5:] for line in result.stdout.splitlines()
                                  if line.startswith("CALL:")], names)

    def test_invalid_mode_stops_before_legacy(self):
        result, _ = self.legacy_trace("invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown SGL_BUILD_MODE", result.stderr)
        self.assertNotIn("CALL:", result.stdout)

    def test_cann_dispatch_skips_legacy_and_propagates_failure(self):
        module = self.root / "python/sgl_kernel_npu/build_tools/cann.py"
        module.parent.mkdir(parents=True)
        module.write_text("import sys\nprint('CANN_ARGS:', repr(sys.argv[1:]))\nsys.exit(7)\n")
        env = dict(os.environ, SGL_BUILD_MODE="CANN")
        result = subprocess.run(["bash", "build.sh", "--soc", "ascend910_93", "--ops", "apply_token_bitmask"],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        self.assertIn("['--soc', 'ascend910_93', '--ops', 'apply_token_bitmask']", result.stdout)
        self.assertNotIn("Output directory:", result.stdout)


if __name__ == "__main__":
    unittest.main()
