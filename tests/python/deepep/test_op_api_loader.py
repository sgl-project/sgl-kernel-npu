"""Exercise the real ELF loader with small shared libraries, without CANN/NPU."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform.startswith("linux"), "Requires the Linux ELF loader")
class TestOpApiLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("g++")
        if cls.compiler is None:
            raise RuntimeError("Install g++ to run the DeepEP loader regression")
        cls.temporary = tempfile.TemporaryDirectory(prefix="deepep-loader-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.external = cls.root / "external"
        cls.external.mkdir()
        cls.build(
            cls.external / "libcust_opapi.so",
            'extern "C" int aclnnDispatchFFNCombine() { return 11; }'
            'extern "C" int aclnnDispatchFFNCombineGetWorkspaceSize() { return 12; }',
            "-Wl,-soname,libcust_opapi.so",
        )
        cls.build(
            cls.external / "libopapi.so",
            'extern "C" int system_only() { return 33; }',
        )
        cls.package = cls.root / "package"
        cls.package.mkdir()
        cls.extension = cls.package / "deep_ep_cpp.so"
        cls.build(
            cls.extension,
            r"""
#include "op_api_loader.hpp"
extern "C" __attribute__((visibility("default"))) int probe(const char *name) {
    auto address = deep_ep::op_api::FindFunction(name);
    return address == nullptr ? -1 : reinterpret_cast<int (*)()>(address)();
}
extern "C" __attribute__((visibility("default"))) const char *custom_path() {
    return deep_ep::op_api::CustomLibrary().path.c_str();
}
extern "C" __attribute__((visibility("default"))) const char *custom_status() {
    static auto status = deep_ep::op_api::Describe(deep_ep::op_api::CustomLibrary());
    return status.c_str();
}
""",
            "-fvisibility=hidden",
        )

    @classmethod
    def build(cls, target, source, *flags):
        source_path = target.with_suffix(".cpp")
        source_path.write_text(source)
        include_dir = Path(__file__).resolve().parents[3] / "csrc" / "deepep"
        subprocess.run(
            [
                cls.compiler,
                "-std=c++17",
                "-shared",
                "-fPIC",
                *flags,
                "-I",
                str(include_dir),
                str(source_path),
                "-o",
                str(target),
                "-ldl",
            ],
            check=True,
        )

    def setUp(self):
        self.case = self.root / self._testMethodName
        self.case.mkdir()
        shutil.copy2(self.extension, self.case / self.extension.name)
        self.bundled = self.case / "vendors/hwcomputing/op_api/lib/libcust_opapi.so"
        self.bundled.parent.mkdir(parents=True)

    def probe(self, *, external=True, preload=False, late_path=False):
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        env.pop("LD_LIBRARY_PATH", None)
        if external:
            env["LD_LIBRARY_PATH"] = str(self.external)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import ctypes, json, os, sys
if sys.argv[3] == 'True':
    old = ctypes.CDLL(sys.argv[2] + '/libcust_opapi.so', mode=ctypes.RTLD_GLOBAL)
if sys.argv[4] == 'True':
    os.environ['LD_LIBRARY_PATH'] = sys.argv[2]
lib = ctypes.CDLL(sys.argv[1])
lib.probe.argtypes = [ctypes.c_char_p]
lib.probe.restype = ctypes.c_int
lib.custom_path.restype = lib.custom_status.restype = ctypes.c_char_p
print(json.dumps({
    'custom': lib.probe(b'aclnnDispatchFFNCombine'),
    'workspace': lib.probe(b'aclnnDispatchFFNCombineGetWorkspaceSize'),
    'system': lib.probe(b'system_only'),
    'missing': lib.probe(b'no_such_function'),
    'path': lib.custom_path().decode(),
    'status': lib.custom_status().decode(),
}))
""",
                str(self.case / self.extension.name),
                str(self.external),
                str(preload),
                str(late_path),
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def build_bundled(self):
        self.build(
            self.bundled,
            'extern "C" int aclnnDispatchFFNCombine() { return 22; }'
            'extern "C" int aclnnDispatchFFNCombineGetWorkspaceSize() { return 23; }',
            "-Wl,-soname,libcust_opapi.so",
        )

    def test_bundled_library_wins_over_old_search_path(self):
        self.build_bundled()
        result = self.probe()
        self.assertEqual(result["custom"], 22)
        self.assertEqual(result["workspace"], 23)
        self.assertEqual(result["system"], 33)
        self.assertEqual(result["missing"], -1)
        self.assertEqual(result["path"], str(self.bundled))

    def test_bundled_library_wins_over_already_loaded_old_library(self):
        self.build_bundled()
        result = self.probe(preload=True)
        self.assertEqual(result["custom"], 22)
        self.assertEqual(result["workspace"], 23)

    def test_bundled_library_does_not_need_startup_search_path(self):
        self.build_bundled()
        self.assertEqual(self.probe(external=False, late_path=True)["custom"], 22)

    def test_external_library_is_supported_for_unbundled_build(self):
        result = self.probe()
        self.assertEqual(result["custom"], 11)
        self.assertEqual(result["workspace"], 12)
        self.assertEqual(result["path"], "libcust_opapi.so")

    def test_broken_bundled_library_does_not_fall_back_to_old_custom_library(self):
        self.bundled.write_text("not an ELF library")
        result = self.probe()
        self.assertEqual(result["custom"], -1)
        self.assertEqual(result["workspace"], -1)
        self.assertEqual(result["system"], 33)
        self.assertIn(str(self.bundled), result["status"])
        self.assertIn("load failed:", result["status"])


if __name__ == "__main__":
    unittest.main()
