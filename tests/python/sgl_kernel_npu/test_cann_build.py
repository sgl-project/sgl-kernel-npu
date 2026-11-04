"""Host-only CANN delivery checks; run directly with Python, without Torch/NPU."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "python/sgl_kernel_npu"))

from build_tools import cann
from build_tools.torch_wheel import stage_runtime, verify_wheel, wheel_build_tag


class CannBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "build/cann"
        self.installed = self.build / "installed"
        self.sdk = self.root / "sdk"
        self.sdk.mkdir()

    def write(self, path, content=b"fixture\n"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_cache(self):
        self.write(
            self.build / "CMakeCache.txt",
            "".join(
                f"{key}:INTERNAL={value}\n" for key, value in self.cache.items()
            ).encode(),
        )

    def fixture(self, *, aicore, aicpu, arch="2201", graph=None):
        # These are payload fixtures, not ELF libraries or compiled Kernels.
        if self.build.exists():
            shutil.rmtree(self.build)
        self.vendor = self.installed / "vendors/sgl_kernel_npu"
        graph = aicore if graph is None else graph
        self.units = cann.architecture(arch)["compute_units"]
        self.pairs = []
        self.headers = []
        for enabled, relative in (
            (aicore, "autogen/aclnn_generated.h"),
            (aicpu, "declared/aclnn_manual.h"),
        ):
            if enabled:
                source = self.write(self.build / relative)
                self.headers.append(source)
                self.pairs.append(
                    (source, self.vendor / "op_api/include" / source.name)
                )
        self.libraries = [self.write(self.vendor / "op_api/lib/libcust_opapi.so")]
        if graph:
            self.libraries.append(
                self.write(self.vendor / "op_proto/lib/libcust_opsproto_rt2.0.so")
            )
        if aicore:
            self.libraries.append(
                self.write(
                    self.vendor
                    / "op_impl/ai_core/tbe/op_tiling/lib/libcust_opmaster_rt2.0.so"
                )
            )
            binary = self.build / "sgl_cann_kernels/binary"
            tbe = self.vendor / "op_impl/ai_core/tbe"
            for unit in self.units:
                for relative in (
                    f"{unit}/op/kernel.o",
                    f"{unit}/op/kernel.json",
                    f"config/{unit}/ops.json",
                ):
                    source = self.write(binary / relative)
                    self.pairs.append((source, tbe / "kernel" / relative))
                info = f"{unit}/aic-{unit}-ops-info.json"
                source = self.write(
                    self.build / "sgl_cann_kernels/tbe/op_info_cfg/ai_core" / info
                )
                self.pairs.append((source, tbe / "config" / info))
        if aicpu:
            for name, destination in (
                ("cust_aicpu_kernel.json", "config"),
                ("libsgl_kernel_npu_aicpu.so", "aicpu_kernel/impl"),
            ):
                source = self.write(self.build / "aicpu" / name)
                self.pairs.append(
                    (source, self.vendor / "op_impl/cpu" / destination / name)
                )
        for source, destination in self.pairs:
            self.write(destination, source.read_bytes())
        self.write(self.vendor / "bin/set_env.bash")
        self.cache = {
            "SGL_CANN_HOST_LIBRARIES": "sgl_cann_opapi"
            + (";sgl_cann_proto" if graph else ""),
            "SGL_CANN_ACLNN_HEADERS": ";".join(map(str, self.headers)),
            "SGL_CANN_AICPU_OPERATORS": "add_example" if aicpu else "",
        }
        self.write_cache()

    def inspect(self, *, aicore, aicpu):
        return cann.inspect_installation(
            self.installed, self.build, self.units, aicore=aicore, aicpu=aicpu
        )

    def test_deliveries_for_both_architectures(self):
        for arch in ("2201", "3510"):
            for aicore, aicpu in ((True, False), (False, True), (True, True)):
                with self.subTest(arch=arch, aicore=aicore, aicpu=aicpu):
                    self.fixture(arch=arch, aicore=aicore, aicpu=aicpu)
                    vendor, libraries, files = self.inspect(aicore=aicore, aicpu=aicpu)
                    self.assertEqual(vendor, self.vendor)
                    self.assertEqual(set(libraries), set(self.libraries))
                    self.assertEqual(len(files), len(set(files)))
                    if not aicore:
                        self.assertFalse((self.build / "autogen").exists())
                        self.assertFalse((vendor / "op_impl/ai_core").exists())

    def test_cpu_graph_is_required_when_declared(self):
        self.fixture(aicore=False, aicpu=True, graph=True)
        self.inspect(aicore=False, aicpu=True)
        self.libraries[-1].unlink()
        with self.assertRaisesRegex(
            RuntimeError, "Expected one installed libcust_opsproto"
        ):
            self.inspect(aicore=False, aicpu=True)

    def test_missing_or_empty_required_host_files_fail(self):
        for index in range(4):
            for empty in (False, True):
                with self.subTest(index=index, empty=empty):
                    self.fixture(aicore=True, aicpu=True)
                    path = [*self.libraries, self.vendor / "bin/set_env.bash"][index]
                    path.write_bytes(b"") if empty else path.unlink()
                    with self.assertRaises(RuntimeError):
                        self.inspect(aicore=True, aicpu=True)

    def test_duplicate_recursive_libraries_fail(self):
        for name in ("libcust_opsproto_rt2.0.so", "libcust_opmaster_rt2.0.so"):
            with self.subTest(name=name):
                self.fixture(aicore=True, aicpu=False)
                path = next(path for path in self.libraries if path.name == name)
                self.write(path.parent / "duplicate" / name)
                with self.assertRaisesRegex(RuntimeError, "Expected one installed"):
                    self.inspect(aicore=True, aicpu=False)

    def test_payload_missing_empty_or_stale_fails(self):
        self.fixture(aicore=True, aicpu=True)
        count = len(self.pairs)
        for index in range(count):
            for failure in ("empty_source", "missing_installed", "stale_installed"):
                with self.subTest(index=index, failure=failure):
                    self.fixture(aicore=True, aicpu=True)
                    source, installed = self.pairs[index]
                    if failure == "empty_source":
                        source.write_bytes(b"")
                    elif failure == "missing_installed":
                        installed.unlink()
                    else:
                        installed.write_bytes(b"different build\n")
                    with self.assertRaises(RuntimeError):
                        self.inspect(aicore=True, aicpu=True)

    def test_missing_declared_payload_fails(self):
        self.fixture(aicore=True, aicpu=True)
        declared = [
            index
            for index, (source, _) in enumerate(self.pairs)
            if not source.is_relative_to(self.build / "sgl_cann_kernels/binary")
        ]
        for index in declared:
            with self.subTest(index=index):
                self.fixture(aicore=True, aicpu=True)
                self.pairs[index][0].unlink()
                with self.assertRaisesRegex(
                    RuntimeError, "Missing or empty build artifact"
                ):
                    self.inspect(aicore=True, aicpu=True)

    def test_external_installed_payload_fails(self):
        self.fixture(aicore=False, aicpu=True)
        source, installed = self.pairs[-1]
        outside = self.write(self.root / "outside.so", source.read_bytes())
        installed.unlink()
        installed.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "external installed artifact"):
            self.inspect(aicore=False, aicpu=True)

    def test_exported_expectations_cannot_be_missing(self):
        for key in ("SGL_CANN_HOST_LIBRARIES", "SGL_CANN_ACLNN_HEADERS"):
            with self.subTest(key=key):
                self.fixture(aicore=False, aicpu=True)
                del self.cache[key]
                self.write_cache()
                with self.assertRaisesRegex(RuntimeError, key):
                    self.inspect(aicore=False, aicpu=True)

    def test_aclnn_requires_exported_headers(self):
        self.fixture(aicore=False, aicpu=True)
        self.cache["SGL_CANN_ACLNN_HEADERS"] = ""
        self.write_cache()
        with self.assertRaisesRegex(RuntimeError, "without exporting its headers"):
            self.inspect(aicore=False, aicpu=True)

    def test_core_requires_each_compute_unit(self):
        self.fixture(aicore=True, aicpu=False)
        unit = self.units[-1]
        shutil.rmtree(self.build / "sgl_cann_kernels/binary" / unit)
        with self.assertRaisesRegex(RuntimeError, "Missing compiled Kernel binaries"):
            self.inspect(aicore=True, aicpu=False)

    def test_core_requires_metadata(self):
        self.fixture(aicore=True, aicpu=False)
        for path in (self.build / "sgl_cann_kernels/binary").rglob("*.json"):
            path.unlink()
        with self.assertRaisesRegex(
            RuntimeError, "Missing compiled Kernel binaries or metadata"
        ):
            self.inspect(aicore=True, aicpu=False)

    def test_coordinator_force_and_real_file_checks(self):
        for arch in ("2201", "3510"):
            for aicore, aicpu in ((True, False), (False, True), (True, True)):
                with self.subTest(arch=arch, aicore=aicore, aicpu=aicpu):
                    self.fixture(arch=arch, aicore=aicore, aicpu=aicpu)
                    selected = ("apply_token_bitmask",) if aicore else ()
                    selected += ("add_example",) if aicpu else ()
                    self.cache.update(
                        {
                            "CMAKE_HOME_DIRECTORY": str(PROJECT_ROOT / "csrc"),
                            "SGL_ARCH_VERSION": arch,
                            "SGL_CANN_OPERATORS": ";".join(selected),
                            "SGL_CANN_SDK_ROOT": str(self.sdk.resolve()),
                        }
                    )
                    self.write_cache()
                    self.write(
                        self.build / "selected-operators.txt",
                        ("\n".join(selected) + "\n").encode(),
                    )
                    self.write(self.build / "package-stage/fixture.run")
                    template = self.root / "installation-template"
                    if template.exists():
                        shutil.rmtree(template)
                    shutil.copytree(self.installed, template)
                    commands = []

                    def run_fixture(command, *args, **kwargs):
                        commands.append(list(map(str, command)))
                        if command[0] == "bash":
                            shutil.copytree(
                                template, self.installed, dirs_exist_ok=True
                            )

                    # The fixture supplies installed files; inspection itself is real.
                    with patch.object(
                        cann.platform, "system", return_value="Linux"
                    ), patch.dict(
                        cann.os.environ, {"ASCEND_HOME_PATH": str(self.sdk)}
                    ), patch.object(
                        cann, "run", side_effect=run_fixture
                    ), contextlib.redirect_stdout(
                        io.StringIO()
                    ):
                        result = cann.build_cann(
                            arch, selected, build_root=self.root / "build"
                        )
                    self.assertEqual(result.operators, selected)
                    self.assertEqual(result.vendor, self.vendor)
                    self.assertEqual(len(commands), 4)
                    self.assertIn("-DSGL_CANN_BUILD_STAGE=KERNEL", commands[0])
                    self.assertEqual(
                        commands[1][-2:], ["--target", "sgl_cann_kernel_check"]
                    )
                    self.assertEqual(commands[2][-2:], ["--target", "package"])
                    self.assertEqual("--force" in commands[3], aicpu and not aicore)

    def test_cpu_only_wheel_staging_and_payload_verification(self):
        for arch in ("2201", "3510"):
            with self.subTest(arch=arch):
                self.fixture(arch=arch, aicore=False, aicpu=True)
                package = self.root / "package/sgl_kernel_npu"
                stage_runtime(self.vendor, package, arch_version=arch)
                manifest = {
                    "schema_version": 1,
                    "build_mode": "REFACTOR",
                    "arch_version": arch,
                    "selection_kind": "custom",
                    "operators": ["add_example"],
                    "compute_units": self.units,
                    "third_party": [],
                }
                self.write(
                    package / "_build_manifest.json", json.dumps(manifest).encode()
                )
                self.write(package / "lib/libsgl_kernel_npu.so")
                staged_vendor = package / "_cann/vendors/sgl_kernel_npu"
                self.assertFalse((staged_vendor / "op_impl/ai_core").exists())
                self.assertFalse((staged_vendor / "op_proto").exists())
                wheel = (
                    self.root
                    / f"fixture-0-{wheel_build_tag(manifest)}-cp312-cp312-linux_aarch64.whl"
                )
                cpu_name = "sgl_kernel_npu/_cann/vendors/sgl_kernel_npu/op_impl/cpu/aicpu_kernel/impl/libsgl_kernel_npu_aicpu.so"
                for omit_device in (False, True):
                    with zipfile.ZipFile(wheel, "w") as archive:
                        for path in package.rglob("*"):
                            if path.is_file():
                                name = (
                                    "sgl_kernel_npu/"
                                    + path.relative_to(package).as_posix()
                                )
                                if not (omit_device and name == cpu_name):
                                    archive.write(path, name)
                        archive.writestr(
                            "fixture-0.dist-info/WHEEL", "Root-Is-Purelib: false\n"
                        )
                    if omit_device:
                        with self.assertRaisesRegex(
                            RuntimeError, "Wheel content differs"
                        ):
                            verify_wheel(wheel, package)
                    else:
                        verify_wheel(wheel, package)


if __name__ == "__main__":
    unittest.main()
