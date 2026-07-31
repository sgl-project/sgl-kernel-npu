import importlib.util
import shutil
import sys
from configparser import ConfigParser
from pathlib import Path
from types import ModuleType

import pytest
import setuptools
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "python" / "sgl_kernel_npu"
NORM_ROOT = PACKAGE_ROOT / "sgl_kernel_npu" / "norm"
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


def stage_norm_sources(build_lib: Path) -> Path:
    norm_dir = build_lib / "sgl_kernel_npu" / "norm"
    norm_dir.mkdir(parents=True)
    for filename in ("gemma_rmsnorm.py", "_gemma_rmsnorm_aclnn.py"):
        shutil.copyfile(NORM_ROOT / filename, norm_dir / filename)
    return norm_dir


@pytest.mark.parametrize(
    ("target", "required_call", "excluded_call"),
    [
        ("Ascend910", "torch_npu.npu_gemma_rms_norm", "torch_npu.npu_rms_norm"),
        ("Ascend950", "torch_npu.npu_rms_norm", "torch_npu.npu_gemma_rms_norm"),
    ],
)
def test_wheel_contains_only_target_gemma_implementation(
    wheel_setup, monkeypatch, tmp_path, target, required_call, excluded_call
):
    norm_dir = stage_norm_sources(tmp_path)
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.setenv(wheel_setup.BUILD_TARGET_ENV, target)
    command = wheel_setup.TargetBuildPy(Distribution())
    command.build_lib = str(tmp_path)

    command.run()

    source = (norm_dir / "gemma_rmsnorm.py").read_text(encoding="utf-8")
    assert required_call in source
    assert excluded_call not in source
    assert not (norm_dir / "_gemma_rmsnorm_aclnn.py").exists()


def test_unknown_wheel_target_is_rejected(wheel_setup, monkeypatch, tmp_path):
    stage_norm_sources(tmp_path)
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.setenv(wheel_setup.BUILD_TARGET_ENV, "FutureAscend")
    command = wheel_setup.TargetBuildPy(Distribution())
    command.build_lib = str(tmp_path)

    with pytest.raises(ValueError, match="Unsupported wheel target"):
        command.run()


def test_build_script_exports_canonical_wheel_target():
    source = (PROJECT_ROOT / "build.sh").read_text(encoding="utf-8")

    assert 'SGL_KERNEL_NPU_BUILD_TARGET="Ascend910"' in source
    assert 'SGL_KERNEL_NPU_BUILD_TARGET="Ascend950"' in source
    assert "export SGL_KERNEL_NPU_BUILD_TARGET" in source
    assert "SGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM" not in source


def test_no_runtime_build_target_or_compiled_capability_remains():
    setup_source = SETUP_PATH.read_text(encoding="utf-8")
    gemma_source = (NORM_ROOT / "gemma_rmsnorm.py").read_text(encoding="utf-8")
    extension_source = (PROJECT_ROOT / "csrc" / "pytorch_extensions.cpp").read_text(
        encoding="utf-8"
    )

    assert "_build_target" not in setup_source
    assert not (PACKAGE_ROOT / "sgl_kernel_npu" / "_build_target.py").exists()
    assert "sgl_kernel_npu_use_native_gemma_rms_norm" not in gemma_source
    assert "sgl_kernel_npu_use_native_gemma_rms_norm" not in extension_source
