"""CPU-only checks for preparing the fixed pre-migration LIM source."""

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from benchmark.lim_a5_mtp_ab.prepare_original import (
    UPSTREAM_REVISION,
    _extract_source,
    prepare,
    safe_archive_path,
)


class OriginalSourcePrepareTests(unittest.TestCase):
    def test_archive_path_rejects_traversal_and_windows_absolute_paths(self):
        for value in ("../escape", "/absolute", "C:/absolute", "csrc/../../escape"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_archive_path(value)

    def test_extract_rejects_non_regular_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "unsafe.tar.gz"
            info = tarfile.TarInfo("csrc/escape-link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.addfile(info)
            with self.assertRaisesRegex(ValueError, "non-regular"):
                _extract_source(archive_path, root / "source")

    def test_prepare_materializes_pinned_limbuild_and_dependency_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            manifest_path = prepare(output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["upstream"]["revision"], UPSTREAM_REVISION)
            self.assertEqual(manifest["operator"], "lim")
            self.assertTrue((output / "source/csrc/build.sh").is_file())
            self.assertTrue(
                (
                    output
                    / "source/csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/arch35/lim_simt_config.h"
                ).is_file()
            )
            self.assertTrue((output / "source/LICENSE").is_file())
            self.assertTrue(
                (
                    output / "source/csrc/third_party/pkg/abseil-cpp-20230802.1.tar.gz"
                ).is_file()
            )
            self.assertTrue(
                (output / "source/csrc/third_party/pkg/protobuf-25.1.tar.gz").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "source/csrc/third_party/licenses/nlohmann-json-LICENSE.MIT"
                ).is_file()
            )
            self.assertEqual(len(manifest["files"]), 401)
            self.assertEqual(len(manifest["packaging_shims"]), 2)
            self.assertTrue(manifest["vendored_dependencies"])

    def test_prepare_refuses_existing_output_root_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare(output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
