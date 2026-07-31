import importlib.util
from pathlib import Path

import pytest

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
    ("target", "expected"),
    [
        ("910B", "Ascend910B1"),
        ("Ascend910B1", "Ascend910B1"),
        ("910C", "Ascend910_9382"),
        ("Ascend910_9382", "Ascend910_9382"),
        ("950", "Ascend950"),
        ("Ascend950", "Ascend950"),
    ],
)
def test_soc_version_aliases_are_normalized(target, expected):
    assert build_targets.normalize_soc_version(target) == expected


def test_unknown_soc_version_is_rejected():
    with pytest.raises(ValueError, match="Unsupported SOC_VERSION"):
        build_targets.normalize_soc_version("FutureAscend")


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("Ascend910B1", "native"),
        ("Ascend910_9382", "native"),
        ("Ascend950", "triton"),
    ],
)
def test_gemma_provider_is_selected_from_build_target(target, expected):
    assert build_targets.get_gemma_provider(target) == expected


@pytest.mark.parametrize(
    ("target", "soc_version", "provider"),
    [
        ("910B", "Ascend910B1", "native"),
        ("910C", "Ascend910_9382", "native"),
        ("950", "Ascend950", "triton"),
    ],
)
def test_build_writes_target_specific_package_config(
    tmp_path, target, soc_version, provider
):
    build_targets.write_build_target_config(tmp_path, target)

    config_path = tmp_path / "sgl_kernel_npu" / "_build_target.py"
    namespace = {}
    exec(config_path.read_text(encoding="utf-8"), namespace)

    assert namespace["SOC_VERSION"] == soc_version
    assert namespace["GEMMA_RMS_NORM_PROVIDER"] == provider


def test_build_target_uses_environment(monkeypatch):
    monkeypatch.delenv(build_targets.BUILD_TARGET_ENV, raising=False)
    assert build_targets.get_build_target() == "Ascend910_9382"

    monkeypatch.setenv(build_targets.BUILD_TARGET_ENV, "950")
    assert build_targets.get_build_target() == "Ascend950"


def test_gemma_public_module_has_no_runtime_soc_dispatch():
    source = (
        MODULE_PATH.parent / "sgl_kernel_npu" / "norm" / "gemma_rmsnorm.py"
    ).read_text(encoding="utf-8")

    assert "get_soc_version" not in source
    assert "NpuDeviceFamily" not in source


def test_build_script_uses_one_normalized_soc_version():
    source = (MODULE_PATH.parents[2] / "build.sh").read_text(encoding="utf-8")

    assert "PRODUCT_TARGET" not in source
    assert "CANN_SOC_VERSION" not in source
    assert '-DSOC_VERSION="$SOC_VERSION"' in source
