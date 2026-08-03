import importlib.util
import shlex
import shutil
import subprocess
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

# npu-smi reports A5 from the device-level query, while A2/A3 need a chip id.
FAKE_NPU_SMI = {
    "a5": 'echo "  Chip Name  : Ascend950PR"',
    "a3": 'if [[ "$*" == *"-c 0"* ]]; then echo "  Chip Name  : Ascend910"; fi',
    "a2": 'if [[ "$*" == *"-c 0"* ]]; then echo "  Chip Name  : 910B4"; fi',
    "unknown": 'echo "  Chip Name  : Ascend310P3"',
}


# On Windows `bash` resolves to WSL, which receives the repo path untranslated
# and whose mktemp/stdout handling breaks under an inherited Windows environment.
# These cases exercise build.sh itself, so they belong to the POSIX CI job.
needs_posix_shell = pytest.mark.skipif(
    sys.platform == "win32",
    reason="needs a POSIX shell sharing the repo path; Windows bash is WSL",
)


def resolve_soc(target, chip=None, requested_soc=""):
    """Run build.sh's SoC resolution with npu-smi faked to report ``chip``.

    The stub is written from inside the shell rather than by the test, so no
    host path ever has to survive translation into whichever bash is on PATH.
    """
    stub = ""
    if chip is not None:
        stub = f"""
        printf '%s\\n' '#!/usr/bin/env bash' {shlex.quote(FAKE_NPU_SMI[chip])} \\
            > "$fake_bin/npu-smi"
        chmod +x "$fake_bin/npu-smi"
        """

    script = f"""
        set -e
        fake_bin="$(mktemp -d)"
        {stub}
        export PATH="$fake_bin:/usr/bin:/bin"
        set +e
        source ./build.sh
        BUILD_TARGET={target}
        REQUESTED_SOC_VERSION="{requested_soc}"
        BUILD_DEEPEP_MODULE=ON
        BUILD_KERNELS_MODULE=ON
        configure_soc_version
    """
    return subprocess.run(
        ["bash", "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


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
    for filename in ("_gemma_rmsnorm_native.py", "_gemma_rmsnorm_aclnn.py"):
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
    assert not (norm_dir / "_gemma_rmsnorm_native.py").exists()
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


@pytest.mark.parametrize(
    ("chip", "wheel_target", "cmake_soc"),
    [
        ("a5", "Ascend950", "Ascend910_9382"),
        ("a3", "Ascend910", "Ascend910_9382"),
        ("a2", "Ascend910", "Ascend910B1"),
        # npu-smi present but the chip is unrecognized, and no npu-smi at all:
        # both keep the A3 target that used to be the unconditional default,
        # so a build that never failed before still cannot fail here.
        ("unknown", "Ascend910", "Ascend910_9382"),
        (None, "Ascend910", "Ascend910_9382"),
    ],
)
@needs_posix_shell
def test_kernels_build_detects_the_local_soc(chip, wheel_target, cmake_soc):
    done = resolve_soc("kernels", chip=chip)

    assert done.returncode == 0, done.stderr
    assert f"Wheel SOC_VERSION: {wheel_target}" in done.stdout
    assert f"CMake SOC_VERSION: {cmake_soc}" in done.stdout


@needs_posix_shell
def test_explicit_soc_version_skips_detection():
    """An explicit argument must win over whatever the local NPU reports."""
    done = resolve_soc("kernels", chip="a5", requested_soc="910B")

    assert done.returncode == 0, done.stderr
    assert "Wheel SOC_VERSION: Ascend910" in done.stdout
    assert "CMake SOC_VERSION: Ascend910B1" in done.stdout


@needs_posix_shell
def test_deepep_still_refuses_an_unrecognized_chip():
    """deepep picks its ops variant from the SoC, so it must not guess."""
    done = resolve_soc("deepep", chip="unknown")

    assert done.returncode != 0
    assert "Cannot determine the device type" in done.stdout + done.stderr


def test_source_tree_ships_no_staged_gemma_provider():
    """The staged name must not exist in the source tree.

    Re-adding ``norm/gemma_rmsnorm.py`` would make ``TargetBuildPy`` a no-op for
    whichever target it happens to match, so an A5 wheel would silently carry
    the 910 operator instead of failing the build.
    """
    assert not (NORM_ROOT / "gemma_rmsnorm.py").exists()
    assert (NORM_ROOT / "_gemma_rmsnorm_native.py").exists()
    assert (NORM_ROOT / "_gemma_rmsnorm_aclnn.py").exists()
