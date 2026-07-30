import importlib.util
from pathlib import Path

import pytest
from setuptools.command.build_py import build_py

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sgl_kernel_npu"
    / "build_targets.py"
)
SPEC = importlib.util.spec_from_file_location("build_targets", MODULE_PATH)
build_targets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_targets)


@pytest.mark.parametrize(
    ("module", "target", "enabled"),
    [
        ("sgl_kernel_npu.norm.gemma_rmsnorm", "910B", False),
        ("sgl_kernel_npu.norm.gemma_rmsnorm", "910C", False),
        ("sgl_kernel_npu.norm.gemma_rmsnorm", "950", True),
        ("sgl_kernel_npu.norm.gemma_rmsnorm", "FutureAscend", False),
        ("sgl_kernel_npu.norm._gemma_rmsnorm_triton", "910B", False),
        ("sgl_kernel_npu.norm._gemma_rmsnorm_triton", "950", True),
    ],
)
def test_gemma_rmsnorm_is_only_packaged_for_ascend950(module, target, enabled):
    assert build_targets.module_is_enabled(module, target) is enabled


def test_unrestricted_modules_are_packaged_for_every_target():
    assert build_targets.module_is_enabled(
        "sgl_kernel_npu.norm.add_rmsnorm_bias", "FutureAscend"
    )


def test_build_target_uses_environment(monkeypatch):
    monkeypatch.delenv(build_targets.BUILD_TARGET_ENV, raising=False)
    assert build_targets.get_build_target() == "910C"

    monkeypatch.setenv(build_targets.BUILD_TARGET_ENV, "950")
    assert build_targets.get_build_target() == "950"


def test_build_py_filters_target_specific_modules(monkeypatch):
    modules = [
        ("sgl_kernel_npu.norm", "gemma_rmsnorm", "gemma_rmsnorm.py"),
        (
            "sgl_kernel_npu.norm",
            "_gemma_rmsnorm_triton",
            "_gemma_rmsnorm_triton.py",
        ),
        ("sgl_kernel_npu.norm", "add_rmsnorm_bias", "add_rmsnorm_bias.py"),
    ]
    monkeypatch.setattr(build_py, "find_package_modules", lambda *_: modules)
    monkeypatch.setenv(build_targets.BUILD_TARGET_ENV, "910C")

    command = object.__new__(build_targets.TargetBuildPy)
    selected = command.find_package_modules("sgl_kernel_npu.norm", "unused")

    assert selected == [modules[2]]


def test_gemma_public_module_has_no_runtime_soc_dispatch():
    source = (
        MODULE_PATH.parent / "sgl_kernel_npu" / "norm" / "gemma_rmsnorm.py"
    ).read_text(encoding="utf-8")

    assert "torch_npu" not in source
    assert "get_soc_version" not in source
    assert "PROVIDERS" not in source


def test_build_target_manifest_uses_product_names():
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "Ascend910" not in source
    assert "Ascend950" not in source
