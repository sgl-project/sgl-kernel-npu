"""Host-only checks for opt-in ZeRCMoE build staging."""

import importlib.metadata
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "python/sgl_kernel_npu"))
    from build_tools import third_party

    spec = importlib.util.spec_from_file_location(
        "zerc_moe_adapter_test", ROOT / "third_party/zerc_moe/build.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return third_party, adapter


def _profile(name, *, opt_in=False):
    return {
        "name": name,
        "opt_in": opt_in,
        "ops": {"arch22": [], "arch35": ["zerc_moe"]},
        "repository": "https://example.invalid/source.git",
        "commit": "1" * 40,
        "builder": name,
        "license": "LICENSE",
    }


def test_opt_in_is_not_selected_by_default(modules, tmp_path, monkeypatch):
    third_party, _ = modules
    profiles = tmp_path / "third_party/profiles/cann-9.2.0"
    profiles.mkdir(parents=True)
    (profiles / "existing.json").write_text(json.dumps(_profile("existing")))
    (profiles / "zerc_moe.json").write_text(
        json.dumps(_profile("zerc_moe", opt_in=True))
    )
    adapter = SimpleNamespace(configure=lambda config, jobs: config)
    monkeypatch.setattr(third_party, "get_adapter", lambda name: adapter)
    default = third_party.resolve_plan("9.2.0", "3510", source_root=tmp_path)
    assert [item["name"] for item in default] == ["existing"]
    selected = third_party.resolve_plan(
        "9.2.0", "3510", source_root=tmp_path, third_party=("zerc_moe",)
    )
    assert [item["name"] for item in selected] == ["zerc_moe"]
    with pytest.raises(ValueError, match="not supported"):
        third_party.resolve_plan(
            "9.2.0", "2201", source_root=tmp_path, third_party=("zerc_moe",)
        )


def test_configure_fingerprints_overlay_and_targets_ascend_950(modules, monkeypatch):
    _, adapter = modules
    monkeypatch.setattr(adapter, "detect_soc_version", lambda: "ascend950")
    state = {"csrc/zerc_moe/source/kernel.cpp": "a"}
    monkeypatch.setattr(adapter, "local_input_hashes", lambda paths: dict(state))
    config = {"arch_version": "3510", "name": "zerc_moe"}
    first = adapter.configure(config, 4)
    assert first["commands"][0][-2:] == ["--ops=zerc_moe", "-j4"]
    state["csrc/zerc_moe/source/kernel.cpp"] = "b"
    assert adapter.configure(config, 4)["overlay_sha256"] != first["overlay_sha256"]
    monkeypatch.setattr(adapter, "detect_soc_version", lambda: "ascend910_93")
    with pytest.raises(RuntimeError, match="Ascend 950"):
        adapter.configure(config, 4)


def test_prepare_source_registers_a_separate_mc2_name(modules, tmp_path, monkeypatch):
    _, adapter = modules
    local = tmp_path / "local"
    local.mkdir()
    (local / "build.py").write_text("def overlay_sources(source):\n    pass\n")
    monkeypatch.setattr(adapter, "_SOURCE_ROOT", local)
    source = tmp_path / "checkout"
    operator = source / "mc2/mega_moe"
    for relative in (
        "op_host/mega_moe_def.cpp",
        "op_kernel/mega_moe.cpp",
        "op_api/aclnn_mega_moe.cpp",
    ):
        file = operator / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(relative)
    (source / "cmake").mkdir()
    (source / "cmake/func.cmake").write_text('        "mega_moe;"\n')
    builder = source / "torch_extension/npu_ops_transformer/op_builder/builder.py"
    builder.parent.mkdir(parents=True)
    builder.write_text('AS_LIBRARY = Library("npu_ops_transformer", "DEF")\n')
    ops_init = source / "torch_extension/npu_ops_transformer/ops/__init__.py"
    ops_init.parent.mkdir(parents=True)
    ops_init.write_text(
        "# Copyright notice\nimport os as _os\n"
        'ENV = "NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE"\n'
        'raise RuntimeError("unrelated eager imports")\n'
    )
    adapter.prepare_source(source)
    isolated = source / "mc2/zerc_moe"
    assert (operator / "op_kernel/mega_moe.cpp").exists()
    assert (isolated / "op_kernel/zerc_moe.cpp").read_text() == "op_kernel/mega_moe.cpp"
    assert (isolated / "op_host/zerc_moe_def.cpp").exists()
    assert (isolated / "op_api/aclnn_zerc_moe.cpp").exists()
    assert '"zerc_moe;"' in (source / "cmake/func.cmake").read_text()
    assert 'Library("zerc_moe", "DEF")' in builder.read_text()
    assert ops_init.read_text().startswith("# Copyright notice\n")
    assert "unrelated eager imports" not in ops_init.read_text()


def test_missing_build_dependency_fails_before_preparing_source(
    modules, tmp_path, monkeypatch
):
    _, adapter = modules
    prepared = []

    def version(name):
        if name == "build":
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(adapter.importlib.metadata, "version", version)
    monkeypatch.setattr(
        adapter, "prepare_source", lambda source: prepared.append(source)
    )
    with pytest.raises(RuntimeError, match="build"):
        adapter.build({}, tmp_path, tmp_path)
    assert not prepared


@pytest.mark.parametrize(
    ("digests", "prepared_count"),
    [(["changed"], 0), (["expected", "changed"], 1)],
)
def test_changed_overlay_aborts_before_build(
    modules, tmp_path, monkeypatch, digests, prepared_count
):
    _, adapter = modules
    prepared = []
    values = iter(digests)
    monkeypatch.setattr(adapter.importlib.metadata, "version", lambda _name: "1.0")
    monkeypatch.setattr(adapter, "_overlay_digest", lambda: next(values))
    monkeypatch.setattr(
        adapter, "prepare_source", lambda source: prepared.append(source)
    )
    with pytest.raises(RuntimeError, match="source changed"):
        adapter.build({"overlay_sha256": "expected"}, tmp_path, tmp_path)
    assert len(prepared) == prepared_count


def test_stage_python_extracts_only_private_bindings(modules, tmp_path):
    _, adapter = modules
    wheel = tmp_path / "source.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for relative in adapter._BINDINGS:
            if relative == "op_builder/builder.py":
                content = (
                    "import npu_ops_transformer\n"
                    'AS_LIBRARY = Library("zerc_moe", "DEF")\n'
                )
            elif relative in ("ops/mega_moe.py", "ops/comm_context.py"):
                content = (
                    "from npu_ops_transformer.op_builder.builder import OpBuilder\n"
                )
            else:
                content = "binding"
            archive.writestr("npu_ops_transformer/" + relative, content)
        archive.writestr("npu_ops_transformer/unrelated.py", "raise RuntimeError()")
    package = tmp_path / "package"
    adapter.stage_python(wheel, package)
    target = package / "_third_party/zerc_moe"
    builder = (target / "op_builder/builder.py").read_text()
    assert 'Library("zerc_moe", "DEF")' in builder
    assert "zerc_moe as npu_ops_transformer" in builder
    assert (
        "from ..op_builder.builder import" in (target / "ops/mega_moe.py").read_text()
    )
    assert not (target / "unrelated.py").exists()


def test_stage_python_rejects_unsafe_wheel(modules, tmp_path):
    _, adapter = modules
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("../../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe wheel"):
        adapter.stage_python(wheel, tmp_path / "package")


def test_inspect_vendor_requires_kernel_metadata_and_binaries(modules, tmp_path):
    _, adapter = modules
    vendor = tmp_path / "vendor"
    library = vendor / "op_api/lib/libcust_opapi.so"
    library.parent.mkdir(parents=True)
    library.write_text("library")
    mc2_libraries = [
        vendor / "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so",
        vendor
        / "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
    ]
    for path in mc2_libraries:
        with pytest.raises(RuntimeError, match="installed MC2 library"):
            adapter.inspect_vendor(vendor, {"soc": "ascend950"})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("library")
    config = vendor / "op_impl/ai_core/tbe/kernel/config/ascend950/zerc_moe.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    binaries = vendor / "op_impl/ai_core/tbe/kernel/ascend950/zerc_moe"
    binaries.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="No ZeRCMoE"):
        adapter.inspect_vendor(vendor, {"soc": "ascend950"})
    (binaries / "kernel.o").write_text("binary")
    (binaries / "kernel.json").write_text("{}")
    adapter.inspect_vendor(vendor, {"soc": "ascend950"})
