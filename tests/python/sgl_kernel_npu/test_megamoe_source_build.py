"""Host-only checks of the standalone MegaMoe build recipe (no CANN build)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "csrc/megamoe/build.py"


@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location("megamoe_build", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_does_not_build_or_create_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", "--work-dir", str(workspace)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--soc=ascend950" in result.stdout
    assert "--ops=mega_moe" in result.stdout
    assert "--vendor_name=sgl_megamoe" in result.stdout
    assert "--wheel --no-isolation" in result.stdout
    assert not workspace.exists()


def test_overlay_uses_repository_source_and_preserves_other_files(
    builder, tmp_path, monkeypatch
):
    local = tmp_path / "local"
    checkout = tmp_path / "checkout"
    for directory in (local, checkout):
        (directory / "mc2/op").mkdir(parents=True)
        (directory / "mc2/op/kernel.cpp").write_text("old")
    (local / "mc2/op/kernel.cpp").write_text("new")
    unrelated = checkout / "mc2/unrelated.cpp"
    unrelated.write_text("keep")
    monkeypatch.setattr(builder, "LOCAL_SOURCE", local)
    monkeypatch.setattr(builder, "OVERLAYS", ("mc2/op",))

    builder.overlay_sources(checkout)

    assert (checkout / "mc2/op/kernel.cpp").read_text() == "new"
    assert unrelated.read_text() == "keep"


def test_overlay_rejects_missing_pinned_file(builder, tmp_path, monkeypatch):
    local = tmp_path / "local"
    checkout = tmp_path / "checkout"
    local.mkdir()
    checkout.mkdir()
    (local / "kernel.cpp").write_text("source")
    monkeypatch.setattr(builder, "LOCAL_SOURCE", local)
    monkeypatch.setattr(builder, "OVERLAYS", ("kernel.cpp",))
    with pytest.raises(RuntimeError, match="Pinned framework is missing"):
        builder.overlay_sources(checkout)


def test_overlay_rejects_incomplete_source_directory(builder, tmp_path, monkeypatch):
    local = tmp_path / "local"
    checkout = tmp_path / "checkout"
    (local / "op").mkdir(parents=True)
    (checkout / "op").mkdir(parents=True)
    (checkout / "op/kernel.cpp").write_text("expected source")
    monkeypatch.setattr(builder, "LOCAL_SOURCE", local)
    monkeypatch.setattr(builder, "OVERLAYS", ("op",))
    with pytest.raises(RuntimeError, match="Source file set differs"):
        builder.overlay_sources(checkout)


def test_overlay_accepts_a_new_local_source_file(builder, tmp_path, monkeypatch):
    local = tmp_path / "local"
    checkout = tmp_path / "checkout"
    (local / "op").mkdir(parents=True)
    (checkout / "op").mkdir(parents=True)
    (local / "op/existing.cpp").write_text("updated")
    (local / "op/new.cpp").write_text("added")
    (checkout / "op/existing.cpp").write_text("old")
    monkeypatch.setattr(builder, "LOCAL_SOURCE", local)
    monkeypatch.setattr(builder, "OVERLAYS", ("op",))

    builder.overlay_sources(checkout)

    assert (checkout / "op/existing.cpp").read_text() == "updated"
    assert (checkout / "op/new.cpp").read_text() == "added"


def test_overlay_rejects_symlinked_destination_parent(builder, tmp_path, monkeypatch):
    local = tmp_path / "local"
    checkout = tmp_path / "checkout"
    outside = tmp_path / "outside"
    (local / "op/nested").mkdir(parents=True)
    (checkout / "op").mkdir(parents=True)
    outside.mkdir()
    (local / "op/nested/new.cpp").write_text("source")
    (checkout / "op/nested").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(builder, "LOCAL_SOURCE", local)
    monkeypatch.setattr(builder, "OVERLAYS", ("op",))

    with pytest.raises(RuntimeError, match="Symlinked framework directory"):
        builder.overlay_sources(checkout)
    assert not (outside / "new.cpp").exists()
