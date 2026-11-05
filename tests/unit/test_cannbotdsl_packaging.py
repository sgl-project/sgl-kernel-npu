"""Source wheel contents and isolation from local DSL compiler output."""

import importlib
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu"
RELATIVE = Path("implementation/cannbot-dsl")


@pytest.fixture(autouse=True)
def build_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "python/sgl_kernel_npu"))


def test_runtime_staging_excludes_local_dsl_exports(tmp_path, monkeypatch):
    wheel = importlib.import_module("build_tools.torch_wheel")
    provider = importlib.import_module("build_tools.target_provider")
    monkeypatch.setattr(wheel, "architecture", lambda *args: {"provider": "Ascend950"})
    monkeypatch.setattr(provider, "stage_target_providers", lambda *args: None)
    source_root = tmp_path / "checkout"
    source = (
        source_root / "python/sgl_kernel_npu/sgl_kernel_npu" / RELATIVE / "attention"
    )
    for name in ("_native/old.bin", "_aicpu/old.so", "build-manifest.json", "entry.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    shutil.copy2(ROOT / "config.ini", source_root / "config.ini")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    package = tmp_path / "staged/sgl_kernel_npu"
    wheel.stage_runtime(vendor, package, source_root, "3510")
    destination = package / RELATIVE / "attention"
    assert {p.name for p in destination.iterdir()} == {"entry.py"}
    assert (destination / "entry.py").read_bytes() == (source / "entry.py").read_bytes()


def test_source_wheel_contains_dsl_sources_and_optional_dependency(
    monkeypatch, tmp_path
):
    """Build a real wheel with fixture binaries, never claim device compilation."""
    import email
    import zipfile

    import setuptools
    from packaging.requirements import Requirement

    wheel = importlib.import_module("build_tools.torch_wheel")
    package = tmp_path / "staged/sgl_kernel_npu"
    attention = package / RELATIVE / "attention"
    attention.mkdir(parents=True)
    for folder in (package, package / "implementation", attention.parent, attention):
        (folder / "__init__.py").write_text("")
    shutil.copytree(SOURCE / RELATIVE, attention.parent, dirs_exist_ok=True)
    (package / "lib").mkdir()
    (package / "lib/libsgl_kernel_npu.so").write_bytes(b"not a compiled library")
    manifest = dict(selection_kind="custom", arch_version="3510", operators=["fixture"])
    (package / "_build_manifest.json").write_text(json.dumps(manifest))
    for name in ("torch", "torch_npu"):
        module = ModuleType(name)
        module.__version__ = "2.13.0"
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("SGL_REFACTOR_STAGED_PACKAGE", str(package))
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    wheel.setup_refactor()
    distribution = captured.pop("distclass")(captured)
    monkeypatch.chdir(tmp_path)
    distribution.script_name = "setup.py"
    distribution.script_args = [
        "bdist_wheel",
        "--dist-dir",
        str(tmp_path / "dist"),
        "--build-number",
        wheel.wheel_build_tag(manifest),
    ]
    distribution.parse_command_line()
    distribution.run_commands()
    built = next((tmp_path / "dist").glob("*.whl"))
    wheel.verify_wheel(built, package)
    with zipfile.ZipFile(built) as archive:
        metadata_name = next(
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
    requirements = [Requirement(v) for v in metadata.get_all("Requires-Dist")]
    exact = [r for r in requirements if r.name == "cannbot-dsl" and r.marker is None]
    assert not exact
    extra = [r for r in requirements if r.name == "cannbot-dsl"]
    assert len(extra) == 1 and str(extra[0].specifier) == "==0.8.0"
    assert extra[0].marker.evaluate({"extra": "cannbot_dsl"})
    assert not extra[0].marker.evaluate({"extra": ""})
