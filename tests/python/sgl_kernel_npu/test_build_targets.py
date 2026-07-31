import importlib.util
import sys
from configparser import ConfigParser
from pathlib import Path
from types import ModuleType

import pytest
import setuptools
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "python" / "sgl_kernel_npu"
SETUP_PATH = PACKAGE_ROOT / "setup.py"


@pytest.fixture
def wheel_setup(monkeypatch):
    cpp_extension = ModuleType("torch_npu.utils.cpp_extension")
    cpp_extension.NpuExtension = lambda *args, **kwargs: object()
    torch_npu_utils = ModuleType("torch_npu.utils")
    torch_npu_utils.cpp_extension = cpp_extension
    torch_npu = ModuleType("torch_npu")
    torch_npu.utils = torch_npu_utils

    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "torch_npu.utils", torch_npu_utils)
    monkeypatch.setitem(sys.modules, "torch_npu.utils.cpp_extension", cpp_extension)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    monkeypatch.setattr(ConfigParser, "get", lambda *args, **kwargs: "0.0.0")

    spec = importlib.util.spec_from_file_location("sgl_kernel_npu_setup", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("target", "soc_version", "provider"),
    [
        ("Ascend910", "Ascend910", "native"),
        ("Ascend950", "Ascend950", "aclnn"),
    ],
)
def test_build_writes_target_specific_package_config(
    wheel_setup, tmp_path, target, soc_version, provider
):
    wheel_setup.write_build_target_config(tmp_path, target)

    config_path = tmp_path / "sgl_kernel_npu" / "_build_target.py"
    namespace = {}
    exec(config_path.read_text(encoding="utf-8"), namespace)

    assert namespace["SOC_VERSION"] == soc_version
    assert namespace["GEMMA_RMS_NORM_PROVIDER"] == provider


def test_unknown_wheel_target_is_rejected(wheel_setup, tmp_path):
    with pytest.raises(ValueError, match="Unsupported wheel target"):
        wheel_setup.write_build_target_config(tmp_path, "FutureAscend")


@pytest.mark.parametrize(
    ("target", "provider"),
    [("Ascend910", "native"), ("Ascend950", "aclnn")],
)
def test_build_py_reads_canonical_target_from_environment(
    wheel_setup, monkeypatch, tmp_path, target, provider
):
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.setenv(wheel_setup.BUILD_TARGET_ENV, target)
    command = wheel_setup.TargetBuildPy(Distribution())
    command.build_lib = str(tmp_path)

    command.run()

    namespace = {}
    config_path = tmp_path / "sgl_kernel_npu" / "_build_target.py"
    exec(config_path.read_text(encoding="utf-8"), namespace)
    assert namespace["SOC_VERSION"] == target
    assert namespace["GEMMA_RMS_NORM_PROVIDER"] == provider


def test_build_py_defaults_to_ascend910(wheel_setup, monkeypatch, tmp_path):
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.delenv(wheel_setup.BUILD_TARGET_ENV, raising=False)
    command = wheel_setup.TargetBuildPy(Distribution())
    command.build_lib = str(tmp_path)

    command.run()

    namespace = {}
    config_path = tmp_path / "sgl_kernel_npu" / "_build_target.py"
    exec(config_path.read_text(encoding="utf-8"), namespace)
    assert namespace["SOC_VERSION"] == "Ascend910"
    assert namespace["GEMMA_RMS_NORM_PROVIDER"] == "native"


def test_gemma_public_module_has_no_runtime_soc_dispatch():
    source = (PACKAGE_ROOT / "sgl_kernel_npu" / "norm" / "gemma_rmsnorm.py").read_text(
        encoding="utf-8"
    )

    assert "get_soc_version" not in source
    assert "NpuDeviceFamily" not in source


def test_build_script_uses_one_normalized_soc_version():
    source = (PACKAGE_ROOT.parents[1] / "build.sh").read_text(encoding="utf-8")

    assert "PRODUCT_TARGET" not in source
    assert "CANN_SOC_VERSION" not in source
    assert 'SGL_KERNEL_NPU_BUILD_TARGET="Ascend910"' in source
    assert 'SGL_KERNEL_NPU_BUILD_TARGET="Ascend950"' in source
    assert 'CMAKE_SOC_VERSION="Ascend910_9382"' in source
    assert '"-DSOC_VERSION=$CMAKE_SOC_VERSION"' in source
    assert "Ascend950PR_* | Ascend950DT_*" in source
    assert "known-working 910C compatibility target" in source
