"""CPU-only regressions for staging differently named upstream bindings."""

import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def staged_indexer_and_prolog(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "python/sgl_kernel_npu"))
    from build_tools.bindings import verify
    from build_tools.third_party import get_adapter

    profile = json.loads(
        (
            PROJECT_ROOT / "third_party/profiles/cann-9.2.0/ops_transformer.json"
        ).read_text()
    )
    item = dict(
        profile,
        builder="ops_transformer",
        python_package=profile["name"],
        vendor_name="vendor_" + profile["name"],
        cann_version="9.2.0",
        arch_version="2201",
        ops=[
            name
            for name in profile["ops"]["arch22"]
            if name.startswith(("mla_prolog", "quant_lightning_indexer"))
        ],
    )
    # Physical upstream wheel paths are independent of the profile selections.
    modules = {
        "mla_prolog_v3": ("mla_prolog", "mla_prolog_functional"),
        "quant_lightning_indexer_v2": (
            "quant_lightning_indexer",
            "quant_lightning_indexer_metadata",
        ),
    }
    wheel = tmp_path / "upstream.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for directory, exports in modules.items():
            archive.writestr(
                "cann_ops_transformer_"
                + item["vendor_name"]
                + "/ops/attention/"
                + directory
                + "/__init__.py",
                "\n".join("def {}(): pass".format(name) for name in exports),
            )
    package = tmp_path / "sgl_kernel_npu"
    package.mkdir()
    (package / "__init__.py").touch()
    get_adapter("ops_transformer").stage_python(wheel, item, package)
    (package / "_build_manifest.json").write_text(
        json.dumps(dict(arch_version="2201", third_party=[item]))
    )

    runtime = ModuleType("sgl_kernel_npu")
    runtime.__file__ = str(package / "__init__.py")
    runtime.__path__ = [str(package)]
    # Isolate these temporary packages from installed libraries and other tests.
    for name in tuple(sys.modules):
        if name == "sgl_kernel_npu" or name.startswith("sgl_kernel_npu."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "sgl_kernel_npu", runtime)
    for suffix, directory in (
        ("_third_party", package / "_third_party"),
        ("_third_party." + item["name"], package / "_third_party" / item["name"]),
    ):
        name = "sgl_kernel_npu." + suffix
        module = ModuleType(name)
        module.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, module)
    ops = importlib.import_module(
        "sgl_kernel_npu._third_party." + item["name"] + ".ops"
    )
    for name in tuple(sys.modules):
        if name.startswith(ops.__name__):
            # Let monkeypatch remove newly imported temporary modules on teardown.
            module = sys.modules.pop(name)
            monkeypatch.setitem(sys.modules, name, module)

    registered = {
        "cann_ops_transformer::mla_prolog",
        "cann_ops_transformer::quant_lightning_indexer",
        "cann_ops_transformer::quant_lightning_indexer_metadata",
    }
    torch = ModuleType("torch")
    torch.ops = SimpleNamespace(
        cann_ops_transformer=SimpleNamespace(
            **{name.split("::")[1]: object() for name in registered}
        )
    )
    torch._C = SimpleNamespace(_dispatch_has_kernel=lambda name: name in registered)
    torch._ops = SimpleNamespace(OpOverloadPacket=type("UnusedPacket", (), {}))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return verify, package, ops, registered


def test_staged_build_names_resolve_to_public_apis(staged_indexer_and_prolog):
    verify, package, _, registered = staged_indexer_and_prolog
    report = verify(package)
    assert set(report["bindings"]) == {name.replace("::", ".") for name in registered}
    assert (package / "_binding_manifest.json").is_file()


def test_staged_missing_python_export_fails(staged_indexer_and_prolog, monkeypatch):
    verify, package, ops, _ = staged_indexer_and_prolog
    monkeypatch.delattr(ops, "mla_prolog")
    with pytest.raises(RuntimeError, match=r"missing sgl_ops_transformer\.mla_prolog"):
        verify(package)


def test_staged_missing_prolog_registration_fails(staged_indexer_and_prolog):
    verify, package, _, registered = staged_indexer_and_prolog
    registered.remove("cann_ops_transformer::mla_prolog")
    with pytest.raises(
        RuntimeError, match="no implementation: cann_ops_transformer::mla_prolog"
    ):
        verify(package)
