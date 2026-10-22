"""Portable checks for incomplete or mismatched package installations, without CANN."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


script = Path(__file__).resolve().parents[2] / "scripts/verify_cann_codegen.py"
spec = importlib.util.spec_from_file_location("verify_cann_codegen", script)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class PackageChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "kernel"
        self.install = self.root / "installed"
        self.soc = "ascend910_93"
        self.vendor = self.install / "vendors/sgl_kernel_npu"
        self.tbe = self.vendor / "op_impl/ai_core/tbe"
        self.binary_root = self.build / "sgl_cann_kernels/binary"
        self.binary = self.tbe / "kernel" / self.soc / "sgl_apply_token_bit_mask/kernel.o"
        self.header = self.vendor / "op_api/include/aclnn_sdk_filename.h"

        # Text fixtures verify file selection and fidelity, not ELF loading or SDK output.
        pairs = [
            (self.build / "autogen/aclnn_sdk_filename.h", self.header),
            (self.binary_root / self.soc / "sgl_apply_token_bit_mask/kernel.o", self.binary),
            (self.binary_root / "config" / self.soc / "binary_info_config.json",
             self.tbe / "kernel/config" / self.soc / "binary_info_config.json"),
            (self.build / "sgl_cann_kernels/tbe/op_info_cfg/ai_core" / self.soc / f"aic-{self.soc}-ops-info.json",
             self.tbe / "config" / self.soc / f"aic-{self.soc}-ops-info.json"),
        ]
        for index, (source, installed) in enumerate(pairs):
            for path in (source, installed):
                self.write(path, f"fixture {index}")
        self.api = self.vendor / "op_api/lib/libcust_opapi.so"
        for path in (self.api, self.vendor / "bin/set_env.bash",
                     self.vendor / "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so",
                     self.tbe / "op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"):
            self.write(path, "fixture")

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def inspect(self):
        return verifier.inspect_installation(self.install, self.build, self.soc)

    def test_complete_inventory(self):
        vendor, libraries, artifacts = self.inspect()
        self.assertEqual(vendor, self.vendor)
        self.assertEqual(len(libraries), 3)
        self.assertIn(self.binary, artifacts)
        self.assertIn(self.header, artifacts)

    def test_missing_library(self):
        self.api.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing, empty or external installed file"):
            self.inspect()

    def test_missing_header(self):
        self.header.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or external installed artifact"):
            self.inspect()

    def test_empty_library(self):
        self.api.write_text("")
        with self.assertRaisesRegex(RuntimeError, "Missing, empty or external installed file"):
            self.inspect()

    def test_changed_header(self):
        self.header.write_text("header from another build")
        with self.assertRaisesRegex(RuntimeError, "differs from this build"):
            self.inspect()

    def test_missing_binary(self):
        self.binary.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or external installed artifact"):
            self.inspect()

    def test_changed_binary(self):
        self.binary.write_text("binary from another build")
        with self.assertRaisesRegex(RuntimeError, "differs from this build"):
            self.inspect()

    def test_missing_metadata(self):
        (self.tbe / "kernel/config" / self.soc / "binary_info_config.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or external installed artifact"):
            self.inspect()

    def test_wrong_target(self):
        with self.assertRaisesRegex(RuntimeError, "Missing compiled Kernel binaries or metadata"):
            verifier.inspect_installation(self.install, self.build, "ascend950")

    def test_external_symlink(self):
        self.binary.unlink()
        self.binary.symlink_to(self.binary_root / self.soc / "sgl_apply_token_bit_mask/kernel.o")
        with self.assertRaisesRegex(RuntimeError, "Missing or external installed artifact"):
            self.inspect()

    def test_require_one_nonempty_package(self):
        with self.assertRaisesRegex(RuntimeError, "Expected one nonempty"):
            verifier.find_run_package(self.build)
        package = self.build / "package-stage/custom_opp_linux_aarch64.run"
        self.write(package, "")
        with self.assertRaisesRegex(RuntimeError, "Expected one nonempty"):
            verifier.find_run_package(self.build)
        package.write_text("fixture")
        self.assertEqual(verifier.find_run_package(self.build), package)
        self.write(package.with_name("another.run"), "fixture")
        with self.assertRaisesRegex(RuntimeError, "Expected one nonempty"):
            verifier.find_run_package(self.build)

    def test_build_failure_stops_before_installation(self):
        with patch.object(verifier, "run", side_effect=RuntimeError("build failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                verifier.verify_package(self.build, self.root, self.root, self.soc)
        self.assertEqual(run.call_count, 1)

    def test_kernel_failure_blocks_package_checkpoint(self):
        def fail_kernel(command, *args, **kwargs):
            if "sgl_bitmask_kernel_check" in command:
                raise RuntimeError("kernel failed")

        argv = [str(script), "--compute-unit", self.soc, "--through", "package",
                "--build-dir", str(self.root / "checkpoints")]
        with patch.object(verifier.sys, "argv", argv), \
                patch.object(verifier.platform, "system", return_value="Linux"), \
                patch.dict(verifier.os.environ, {"ASCEND_HOME_PATH": str(self.root)}), \
                patch.object(verifier, "run", side_effect=fail_kernel), \
                patch.object(verifier, "verify_package") as package:
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                verifier.main()
        package.assert_not_called()

    def test_private_install_and_fresh_loader_process(self):
        package = self.build / "package-stage/custom_opp_linux_aarch64.run"
        self.write(package, "fixture")
        run_root = self.root / "new-run"
        run_root.mkdir()
        with patch.object(verifier, "run") as run, \
                patch.object(verifier, "inspect_installation", return_value=self.inspect()):
            verifier.verify_package(self.build, run_root, run_root, self.soc)
        install_call = run.call_args_list[1]
        self.assertIn(f"--install-path={run_root / 'installed'}", install_call.args[0])
        load_call = run.call_args_list[2]
        self.assertEqual(load_call.kwargs["env"]["ASCEND_CUSTOM_OPP_PATH"], str(self.vendor))
        self.assertEqual(load_call.args[0][1], "-c")
        self.assertIn(str(self.api), load_call.args[0])


if __name__ == "__main__":
    unittest.main()
