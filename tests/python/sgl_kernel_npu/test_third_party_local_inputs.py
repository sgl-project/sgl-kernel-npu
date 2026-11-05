"""Validate cache inputs for repository-owned third-party source overlays."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def local_input_hashes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "python/sgl_kernel_npu"))
    from build_tools.local_inputs import local_input_hashes

    return local_input_hashes


def test_local_source_changes_and_additions_change_hashes(local_input_hashes, tmp_path):
    source = tmp_path / "csrc/zerc_moe/source"
    source.mkdir(parents=True)
    kernel = source / "kernel.cpp"
    kernel.write_text("one")
    inputs = ["csrc/zerc_moe/source"]
    first = local_input_hashes(inputs, tmp_path)
    assert first == local_input_hashes(inputs, tmp_path)
    assert first == {
        "csrc/zerc_moe/source/kernel.cpp": hashlib.sha256(b"one").hexdigest()
    }
    kernel.write_text("two")
    changed = local_input_hashes(inputs, tmp_path)
    assert changed != first
    extra = source / "extra.hpp"
    extra.write_text("new")
    assert local_input_hashes(inputs, tmp_path) != changed
    # The standard cache fingerprint includes the resolved configuration.
    assert hashlib.sha256(json.dumps(first, sort_keys=True).encode()).digest() != (
        hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).digest()
    )


@pytest.mark.parametrize(
    "path", ["/outside", "../outside", "nested/../outside", ".", r"..\outside"]
)
def test_local_input_rejects_external_or_ambiguous_paths(
    local_input_hashes, tmp_path, path
):
    with pytest.raises(ValueError, match="Invalid local build input"):
        local_input_hashes([path], tmp_path)


def test_local_input_rejects_missing_files_and_symlinks(local_input_hashes, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.cpp").write_text("source")
    with pytest.raises(ValueError, match="Missing local build input"):
        local_input_hashes(["missing"], tmp_path)
    (source / "link").symlink_to(source / "kernel.cpp")
    with pytest.raises(ValueError, match="Symlinked local build input"):
        local_input_hashes(["source"], tmp_path)
    with pytest.raises(ValueError, match="Symlinked local build input"):
        local_input_hashes(["source/link"], tmp_path)
