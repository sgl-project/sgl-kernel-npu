"""Offline checks of DSL build plans, Native resources and packaging failures."""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "python/sgl_kernel_npu"))
    return importlib.import_module("build_tools.cannbotdsl")


@pytest.mark.parametrize("target", ["builder", "ci"])
@pytest.mark.parametrize("size", [0, 17, 1024 * 1024 + 17, 3 * 1024 * 1024 + 19])
def test_file_hash_covers_empty_files_chunks_and_tail(
    backend, monkeypatch, tmp_path, target, size
):
    module = (
        backend
        if target == "builder"
        else load_file("_hash_ci", ROOT / "scripts/check_cannbotdsl_ci.py", monkeypatch)
    )
    data = (bytes(range(256)) * ((size + 255) // 256))[:size]
    path = tmp_path / "operator.so"
    path.write_bytes(data)
    assert module._file_sha256(path) == hashlib.sha256(data).hexdigest()
    if size:
        changed = data[:-1] + bytes([data[-1] ^ 255])
        path.write_bytes(changed)
        assert module._file_sha256(path) == hashlib.sha256(changed).hexdigest()
        assert module._file_sha256(path) != hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "version,arch", [("9.2.1", "3510"), ("9.2.0", "2201"), ("9.2", "3510")]
)
def test_build_rejects_other_target_combinations(backend, version, arch):
    with pytest.raises(ValueError, match="CANN 9.2.0"):
        backend.resolve_plan(version, arch)


def make_bundle(backend, tmp_path):
    output = tmp_path / "bundle"
    files = ["_native/manifest.json", "_native/index.json"]
    files += [f"_native/operators/{name}/operator.json" for name in backend.OPERATORS]
    files += [f"_aicpu/{name}" for _, _, name in backend.METADATA]
    for name in files:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture")
    source = tmp_path / "package" / backend.RELATIVE / "attention"
    source.mkdir(parents=True)
    (source / "entry.py").write_text("# same compiled source\n")
    manifest = dict(
        backend.resolve_plan("9.2.0", "3510"),
        sources=backend._hashes(source),
        resources=backend._hashes(output),
    )
    (output / "build-manifest.json").write_text(json.dumps(manifest))
    return output, source


def test_resources_and_sources_are_verified_before_staging(backend, tmp_path):
    output, source = make_bundle(backend, tmp_path)
    result = backend.stage_backend(output, tmp_path / "package")
    assert result["version"] == "0.8.0"
    for _, _, filename in backend.METADATA:
        assert (source / "_aicpu" / filename).read_bytes() == b"test fixture"


@pytest.mark.parametrize(
    "field,value",
    [("target", "dav-2201"), ("operators", ["qli"]), ("backend", "other")],
)
def test_bundle_target_contract_is_checked(backend, tmp_path, field, value):
    output, source = make_bundle(backend, tmp_path)
    path = output / "build-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="bundle contract"):
        backend.stage_backend(output, tmp_path / "package")
    assert not (source / "_native").exists()


@pytest.mark.parametrize("missing", ["torch", "torch_npu"])
def test_missing_torch_dependency_fails_before_build_work(
    backend, tmp_path, monkeypatch, missing
):
    third_party = importlib.import_module("build_tools.third_party")
    monkeypatch.setattr(third_party, "detect_cann_version", lambda: "9.2.0")
    monkeypatch.setattr(backend.sys, "platform", "linux")
    monkeypatch.setattr(backend.sys, "version_info", (3, 12, 0))

    def version(name):
        if name == missing:
            raise importlib.metadata.PackageNotFoundError(name)
        return {"cannbot-dsl": "0.8.0", "torch": "2.10.0", "torch_npu": "2.10.0"}[name]

    monkeypatch.setattr(backend.importlib.metadata, "version", version)
    monkeypatch.setattr(backend, "run", lambda *a, **k: pytest.fail("compiler started"))
    with pytest.raises(RuntimeError, match=missing):
        backend.build_backend(source_root=tmp_path)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("change", ["resource", "source", "missing"])
def test_incomplete_or_stale_bundle_is_rejected(backend, tmp_path, change):
    output, source = make_bundle(backend, tmp_path)
    if change == "resource":
        (output / "_aicpu/metadata_kernel.so").write_bytes(b"corrupt")
    elif change == "source":
        (source / "entry.py").write_text("# edited after compilation\n")
    else:
        (output / "_aicpu/metadata_kernel.so").unlink()
    with pytest.raises(RuntimeError, match="checksum|sources changed"):
        backend.stage_backend(output, tmp_path / "package")
    assert not (source / "_native").exists()


def test_compile_worker_uses_separate_aicpu_workdirs(backend, tmp_path, monkeypatch):
    source, output = tmp_path / "sources", tmp_path / "output"
    source.mkdir()
    (source / "__init__.py").write_text("")
    for name, kernel, _ in backend.METADATA:
        (source / f"{name}.py").write_text(f"def {kernel}(): pass\n")
    compiled, exports = [], []
    aot = ModuleType("cannbotdsl.aot")
    aot.build = lambda **kwargs: exports.append(kwargs)
    toolchain = ModuleType("cannbotdsl.aicpu.toolchain")

    def compile_aicpu(kernel, **kwargs):
        path = Path(kwargs["workdir"])
        path.mkdir(parents=True)
        library = path / "kernel.so"
        library.write_bytes(kernel.__module__.encode())
        compiled.append(kwargs)
        return SimpleNamespace(so_path=library)

    toolchain.compile_aicpu_kernel = compile_aicpu
    monkeypatch.setitem(sys.modules, "cannbotdsl.aot", aot)
    monkeypatch.setitem(sys.modules, "cannbotdsl.aicpu.toolchain", toolchain)
    try:
        backend.compile_sources(source, output)
    finally:
        for key in list(sys.modules):
            if key.startswith("_sgl_ds41_metadata"):
                sys.modules.pop(key)
    assert exports == [
        dict(
            collectors=source.resolve(),
            operators=list(backend.OPERATORS),
            target="dav-3510",
            output=output.resolve() / "_native",
        )
    ]
    assert len({item["workdir"] for item in compiled}) == 3
    assert all(
        item["launch_mode"] == "interface" and item["npu_arch"] == "dav-3510"
        for item in compiled
    )
    assert len(list((output / "_aicpu").glob("*.so"))) == 3


def test_cli_plan_and_wheel_tag_include_dsl(backend, monkeypatch, capsys):
    setup = load_file("_dsl_test_setup", ROOT / "setup.py", monkeypatch)
    setup.main(
        [
            "--arch-version",
            "3510",
            "--without-third-party",
            "--with-cannbot-dsl",
            "--plan",
            "--cann-version",
            "9.2.0",
        ]
    )
    assert json.loads(capsys.readouterr().out) == [
        backend.resolve_plan("9.2.0", "3510")
    ]
    wheel = importlib.import_module("build_tools.torch_wheel")
    base = dict(selection_kind="all", arch_version="3510", operators=[], third_party=[])
    assert wheel.wheel_build_tag(base) == "1_refactor_3510_all"
    native = dict(base, cannbotdsl={"version": "0.8.0", "resources": {"x": "a"}})
    first = wheel.wheel_build_tag(native)
    assert "_cb_" in first
    native["cannbotdsl"]["resources"]["x"] = "b"
    assert wheel.wheel_build_tag(native) != first


@pytest.mark.parametrize(
    "arguments",
    [
        ["--arch-version", "2201", "--with-cannbot-dsl"],
        ["--arch-version", "3510", "--artifact", "run", "--with-cannbot-dsl"],
        ["--arch-version", "3510", "--with-cannbot-dsl", "--cann-version", "9.2.0"],
    ],
)
def test_cli_rejects_invalid_dsl_build_modes(backend, monkeypatch, arguments):
    setup = load_file("_dsl_test_setup", ROOT / "setup.py", monkeypatch)
    with pytest.raises(SystemExit):
        setup.arguments(arguments)


@pytest.mark.parametrize("native", [False, True])
def test_real_wheel_contains_sources_licenses_and_optional_native_assets(
    backend, monkeypatch, tmp_path, native
):
    """Build a real wheel with fixture binaries, never claim device compilation."""
    import email
    import zipfile

    import setuptools
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    wheel = importlib.import_module("build_tools.torch_wheel")
    package = tmp_path / "staged/sgl_kernel_npu"
    attention = package / backend.RELATIVE / "attention"
    attention.mkdir(parents=True)
    for folder in (package, package / "implementation", attention.parent, attention):
        (folder / "__init__.py").write_text("")
    (attention / "entry.py").write_text("# source fixture\n")
    (attention.parent / "LICENSE.cannbot-dsl").write_text("CANN license fixture\n")
    (attention.parent / "UPSTREAM.json").write_text("{}\n")
    (package / "lib").mkdir()
    (package / "lib/libsgl_kernel_npu.so").write_bytes(b"not a compiled library")
    manifest = dict(selection_kind="custom", arch_version="3510", operators=["fixture"])
    if native:
        manifest["cannbotdsl"] = {"version": "0.8.0"}
        (attention / "_native").mkdir()
        (attention / "_native/manifest.json").write_text("{}")
        (attention / "_aicpu").mkdir()
        for _, _, name in backend.METADATA:
            (attention / "_aicpu" / name).write_bytes(b"AICPU fixture")
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
    assert bool(exact) == native
    if native:
        assert str(exact[0].specifier) == "==0.8.0"
        assert SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(">=3.10,<3.13")
