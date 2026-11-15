"""CPU-only contract checks for activating the bundled original A5 OPP."""

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def original_library_scope(opp_root):
    """Temporarily put an isolated reference OPP first in discovery order."""
    before = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if opp_root:
        selected = str(Path(opp_root).resolve(strict=True))
        current = (before or "").split(os.pathsep)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
            dict.fromkeys([selected] + [path for path in current if path])
        )
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
        else:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = before


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "python/sgl_kernel_npu/sgl_kernel_npu/_original_opp.py"
)
REPOSITORY_ROOT = MODULE_PATH.parents[3]
SPEC = importlib.util.spec_from_file_location("_original_opp_under_test", MODULE_PATH)
ORIGINAL_OPP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL_OPP)
STAGE_PATH = Path(__file__).resolve().parents[1] / "tools/a5_c8/stage_runtime.py"
STAGE_SPEC = importlib.util.spec_from_file_location(
    "_stage_original_under_test", STAGE_PATH
)
STAGE = importlib.util.module_from_spec(STAGE_SPEC)
STAGE_SPEC.loader.exec_module(STAGE)
RECORD_SPEC = importlib.util.spec_from_file_location(
    "_record_original_under_test", STAGE_PATH.with_name("record_install.py")
)
RECORD = importlib.util.module_from_spec(RECORD_SPEC)
RECORD_SPEC.loader.exec_module(RECORD)

VENDOR = "vendors/sglc8original_transformer"
REVISIONS = {
    "lim": "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e",
}


class OriginalOppRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._old_custom_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        self._old_ld_library_path = os.environ.get("LD_LIBRARY_PATH")

    def tearDown(self):
        self._restore_env("ASCEND_CUSTOM_OPP_PATH", self._old_custom_opp_path)
        self._restore_env("LD_LIBRARY_PATH", self._old_ld_library_path)

    @staticmethod
    def _restore_env(key, value):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def _make_bundle(self, package_root):
        bundle = package_root / "lib/a5_original_opp"
        vendor_root = bundle / VENDOR
        op_api_lib = vendor_root / "op_api/lib"
        op_api_lib.mkdir(parents=True)
        # This is deliberately just a marker file; no NPU library is loaded.
        library = op_api_lib / "libcust_opapi.so"
        library.write_bytes(b"CPU-only fake shared object")
        runtime_file = vendor_root / "op_impl/ai_core/tbe/config/ascend910b/a5.json"
        runtime_file.parent.mkdir(parents=True)
        runtime_file.write_text("{}\n", encoding="utf-8")
        files = {}
        for path in (library, runtime_file):
            relative = path.relative_to(bundle).as_posix()
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "backend": "original",
            "vendor": VENDOR,
            "files": files,
            "operators": {
                name: {"revision": revision} for name, revision in REVISIONS.items()
            },
        }
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "runtime_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return bundle, manifest

    def test_relocated_bundle_activates_and_preserves_search_path_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            first_root = Path(directory) / "build-root"
            _, manifest = self._make_bundle(first_root)
            # Simulate moving the installed package tree to a different prefix.
            moved_root = Path(directory) / "relocated prefix" / "site-packages"
            moved_root.parent.mkdir()
            first_root.rename(moved_root)
            vendor = str((moved_root / "lib/a5_original_opp" / VENDOR).resolve())
            other = os.pathsep.join(
                ["/existing/one", vendor, "/existing/two", "/existing/one"]
            )
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = other
            os.environ["LD_LIBRARY_PATH"] = "/existing/runtime/libs"

            result = ORIGINAL_OPP.activate(moved_root)

            self.assertEqual(result, manifest)
            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"],
                os.pathsep.join([vendor, "/existing/one", "/existing/two"]),
            )
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/existing/runtime/libs")

    def test_hash_mismatch_and_missing_manifest_file_are_rejected_without_env_change(
        self,
    ):
        for mutation in ("hash", "missing"):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "package"
                bundle, manifest = self._make_bundle(root)
                relative = next(iter(manifest["files"]))
                target = bundle / relative
                if mutation == "hash":
                    target.write_bytes(b"tampered")
                else:
                    target.unlink()
                original = "/existing/custom/opp"
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = original
                with self.assertRaises((ImportError, OSError, ValueError)):
                    ORIGINAL_OPP.activate(root)
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], original)

    def test_manifest_path_escape_is_rejected_without_env_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            bundle, manifest = self._make_bundle(root)
            outside = Path(directory) / "outside.so"
            outside.write_bytes(b"outside")
            relative_escape = Path(os.path.relpath(outside, bundle)).as_posix()
            manifest["files"] = {
                relative_escape: hashlib.sha256(outside.read_bytes()).hexdigest()
            }
            (bundle / "runtime_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            original = "/existing/custom/opp"
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = original

            with self.assertRaises((ImportError, OSError, ValueError)):
                ORIGINAL_OPP.activate(root)

            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], original)

    def test_symlinked_vendor_is_rejected_without_env_change(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "package"
            bundle, manifest = self._make_bundle(root)
            vendor_path = bundle / VENDOR
            real_vendor = base / "real-vendor"
            vendor_path.rename(real_vendor)
            vendor_path.symlink_to(real_vendor, target_is_directory=True)
            original = "/existing/custom/opp"
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = original

            with self.assertRaises((ImportError, OSError, ValueError)):
                ORIGINAL_OPP.activate(root)

            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], original)

    def test_missing_bundle_is_a_noop_for_direct_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            original = "/direct/backend/opp"
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = original

            result = ORIGINAL_OPP.activate(Path(directory))

            self.assertIsNone(result)
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], original)

    def test_runtime_and_build_helpers_keep_python37_syntax_and_apis(self):
        helpers = (
            MODULE_PATH,
            REPOSITORY_ROOT / "tools/a5_c8/stage_runtime.py",
            REPOSITORY_ROOT / "tools/a5_c8/record_install.py",
            REPOSITORY_ROOT / "tools/a5_c8/prepare.py",
            REPOSITORY_ROOT / "benchmark/lim_a5_mtp_ab/prepare_original.py",
        )
        for helper in helpers:
            with self.subTest(helper=helper):
                source = helper.read_text(encoding="utf-8")
                ast.parse(source, filename=str(helper), feature_version=(3, 7))
                self.assertNotIn(".is_relative_to(", source)
        self.assertNotIn(" | None", MODULE_PATH.read_text(encoding="utf-8"))

    def test_only_legacy_a5_build_stages_the_original_lim_runtime(self):
        build = (REPOSITORY_ROOT / "build.sh").read_text(encoding="utf-8")
        cmake = (REPOSITORY_ROOT / "cmake/legacy_kernels.cmake").read_text(
            encoding="utf-8"
        )
        readme = (REPOSITORY_ROOT / "tools/a5_c8/README.md").read_text(encoding="utf-8")
        self.assertIn("SGL_KERNEL_A5_C8_BACKEND:-original", build)
        self.assertIn("tools/a5_c8/stage_runtime.py", build)
        self.assertIn("SGL_KERNEL_A5_C8_ORIGINAL_SOURCE", cmake)
        self.assertIn(
            "fused_quant_lightning_indexer_manage/legacy_registration.cpp", cmake
        )
        self.assertIn(
            "only the legacy `bash build.sh -a kernels ascend950`", readme.lower()
        )
        self.assertIn("does not package the original", readme)

    def test_reference_adapter_load_scope_restores_sglang_vendor_after_failure(self):
        """A separately loaded reference must capture its own vendor first."""
        with tempfile.TemporaryDirectory() as directory:
            original = str(Path(directory).resolve())
            selected = "/wheel/sglang/vendor"
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = selected
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                with original_library_scope(original):
                    self.assertEqual(
                        os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep),
                        [original, selected],
                    )
                    raise RuntimeError("load failed")
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], selected)

    def test_actual_resolved_library_must_be_recorded_and_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            bundle, manifest = self._make_bundle(root)
            recorded = bundle / next(iter(manifest["files"]))
            ORIGINAL_OPP.verify_library(root, manifest, str(recorded))
            unrecorded = recorded.with_name("unrecorded.so")
            unrecorded.write_bytes(b"different runtime")
            with self.assertRaisesRegex(ImportError, "not recorded"):
                ORIGINAL_OPP.verify_library(root, manifest, str(unrecorded))
            recorded.write_bytes(b"changed after activation")
            with self.assertRaisesRegex(ImportError, "SHA256 mismatch"):
                ORIGINAL_OPP.verify_library(root, manifest, str(recorded))

    def test_environment_exporter_runs_without_importing_the_package(self):
        script = MODULE_PATH.parents[3] / "tools/a5_c8/env.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            self._make_bundle(root)
            shutil.copyfile(MODULE_PATH, root / "_original_opp.py")
            result = subprocess.run(
                [sys.executable, str(script), "--package-root", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("export ASCEND_CUSTOM_OPP_PATH=", result.stdout)
            self.assertIn("export LD_LIBRARY_PATH=", result.stdout)
            (root / "lib/a5_original_opp/runtime_manifest.json").unlink()
            failure = subprocess.run(
                [sys.executable, str(script), "--package-root", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(failure.returncode, 2)
            self.assertEqual(failure.stdout, "")

    def _make_build(self, root):
        source = root / "prepared/source/csrc/payload.h"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"pinned upstream source")
        metadata = {
            "operators": {
                name: {"revision": revision} for name, revision in REVISIONS.items()
            },
            "files": [
                {
                    "path": "source/csrc/payload.h",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            ],
        }
        (root / "prepared/SOURCE_MANIFEST.json").write_text(json.dumps(metadata))
        library = root / "opp" / VENDOR / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"CPU-only installed library marker")
        receipt = {
            library.relative_to(root / "opp")
            .as_posix(): hashlib.sha256(library.read_bytes())
            .hexdigest()
        }
        (root / "installed_files.sha256.json").write_text(json.dumps(receipt))
        return source, library

    def test_staged_runtime_survives_relocation_and_direct_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self._make_build(base / "build")
            package = base / "package"
            manifest_path = STAGE.stage(base / "build", package)
            self.assertTrue(manifest_path.is_file())
            relocated = base / "relocated"
            package.rename(relocated)
            manifest = ORIGINAL_OPP.activate(relocated)
            library = (
                relocated
                / "lib/a5_original_opp"
                / VENDOR
                / "op_api/lib/libcust_opapi.so"
            )
            ORIGINAL_OPP.verify_library(relocated, manifest, str(library))
            self.assertEqual(
                manifest["operator_backends"],
                {"lim": "original_aclnn", "copy_sfa": "direct"},
            )
            STAGE.clear(relocated)
            self.assertIsNone(ORIGINAL_OPP.activate(relocated))

    def test_cann_internal_tiling_link_is_recorded_and_survives_relocation(self):
        """CANN installs liboptiling.so as an internal relative symlink."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            build = base / "build"
            self._make_build(build)
            tiling = build / "opp" / VENDOR / "op_impl/ai_core/tbe/op_tiling"
            actual = tiling / "lib/linux/aarch64/libcust_opmaster_rt2.0.so"
            actual.parent.mkdir(parents=True)
            actual.write_bytes(b"CPU-only tiling library marker")
            link = tiling / "liboptiling.so"
            link.symlink_to(Path("lib") / "linux/aarch64/libcust_opmaster_rt2.0.so")
            receipt = RECORD.record(build)
            self.assertEqual(receipt["internal_symlinks"], 1)
            package = base / "package"
            STAGE.stage(build, package)
            relocated = base / "relocated"
            package.rename(relocated)
            copied = (
                relocated
                / "lib/a5_original_opp"
                / VENDOR
                / link.relative_to(build / "opp" / VENDOR)
            )
            self.assertFalse(copied.is_symlink())
            self.assertTrue(link.is_symlink())
            manifest = ORIGINAL_OPP.activate(relocated)
            self.assertTrue(manifest["source_symlinks"])
            ORIGINAL_OPP.verify_library(relocated, manifest, str(copied))

    def test_record_install_rejects_link_outside_its_vendor(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            build = base / "build"
            self._make_build(build)
            outside = base / "outside.so"
            outside.write_bytes(b"outside")
            link = build / "opp" / VENDOR / "outside.so"
            link.symlink_to(os.path.relpath(outside, link.parent))
            with self.assertRaisesRegex(ValueError, "relocatable vendor"):
                RECORD.record(build)

    def test_reusing_changed_source_or_install_is_rejected_before_replacing_package(
        self,
    ):
        for target in ("source", "library"):
            with self.subTest(
                target=target
            ), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                source, library = self._make_build(base / "build")
                package = base / "package"
                manifest_path = STAGE.stage(base / "build", package)
                before = manifest_path.read_bytes()
                (source if target == "source" else library).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    STAGE.stage(base / "build", package)
                self.assertEqual(manifest_path.read_bytes(), before)

    def test_unrecorded_install_file_is_rejected_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self._make_build(base / "build")
            extra = base / "build" / "opp" / VENDOR / "op_api/lib/unrecorded.so"
            extra.write_bytes(b"not in the install receipt")
            with self.assertRaisesRegex(ValueError, "receipt does not match"):
                STAGE.stage(base / "build", base / "package")

    def test_changed_vendored_dependency_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self._make_build(base / "build")
            dependency = base / "build/prepared/source/csrc/third_party/input.tar.gz"
            dependency.parent.mkdir(parents=True)
            dependency.write_bytes(b"pinned dependency")
            manifest_path = base / "build/prepared/SOURCE_MANIFEST.json"
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata["vendored_dependencies"] = [
                {
                    "path": dependency.relative_to(base / "build/prepared").as_posix(),
                    "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
                }
            ]
            manifest_path.write_text(json.dumps(metadata), encoding="utf-8")
            dependency.write_bytes(b"changed dependency")
            with self.assertRaisesRegex(ValueError, "changed"):
                STAGE.stage(base / "build", base / "package")


if __name__ == "__main__":
    unittest.main()
