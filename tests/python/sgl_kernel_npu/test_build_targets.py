import importlib.util
import shlex
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
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("target", "required_call", "excluded_call"),
    [
        ("Ascend910", "torch_npu.npu_gemma_rms_norm", "torch_npu.npu_rms_norm"),
        ("Ascend950", "torch_npu.npu_rms_norm", "torch_npu.npu_gemma_rms_norm"),
    ],
)
def test_wheel_stages_only_target_provider(
    wheel_setup, monkeypatch, tmp_path, target, required_call, excluded_call
):
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.setenv(wheel_setup.BUILD_TARGET_ENV, target)
    command = wheel_setup.TargetBuildPy(Distribution())
    command.build_lib = str(tmp_path)

    command.run()

    staged = tmp_path / "sgl_kernel_npu" / "norm" / "gemma_rmsnorm.py"
    assert staged.exists()
    source = staged.read_text(encoding="utf-8")
    assert required_call in source
    assert excluded_call not in source
    assert not (tmp_path / "target_providers").exists()


def test_unknown_wheel_target_is_rejected(wheel_setup, monkeypatch, tmp_path):
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.setenv(wheel_setup.BUILD_TARGET_ENV, "FutureAscend")
    command = wheel_setup.TargetBuildPy(Distribution())
    command.build_lib = str(tmp_path)

    with pytest.raises(ValueError, match="Unsupported provider target"):
        command.run()


def test_missing_build_target_env_is_rejected(wheel_setup, monkeypatch, tmp_path):
    """A source/editable build must fail loudly.

    Without the env var there is no provider to stage, and quietly
    defaulting to the 910 one would let it run on an A5 host.
    """
    monkeypatch.setattr(build_py, "run", lambda self: None)
    monkeypatch.delenv(wheel_setup.BUILD_TARGET_ENV, raising=False)
    command = wheel_setup.TargetBuildPy(Distribution())
    command.build_lib = str(tmp_path)

    with pytest.raises(RuntimeError, match="must be set when building the wheel"):
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


@pytest.mark.parametrize(
    ("requested_soc", "wheel_target", "cmake_soc"),
    [
        ("910B", "Ascend910", "Ascend910B1"),
        ("950", "Ascend950", "Ascend910_9382"),
        ("Ascend950PR_9599", "Ascend950", "Ascend950PR_9599"),
    ],
)
@needs_posix_shell
def test_explicit_soc_version_skips_detection(requested_soc, wheel_target, cmake_soc):
    """An explicit argument must win over whatever the local NPU reports."""
    done = resolve_soc("kernels", chip="a5", requested_soc=requested_soc)

    assert done.returncode == 0, done.stderr
    assert f"Wheel SOC_VERSION: {wheel_target}" in done.stdout
    assert f"CMake SOC_VERSION: {cmake_soc}" in done.stdout


@needs_posix_shell
def test_deepep_still_refuses_an_unrecognized_chip():
    """deepep picks its ops variant from the SoC, so it must not guess."""
    done = resolve_soc("deepep", chip="unknown")

    assert done.returncode != 0
    assert "Cannot determine the device type" in done.stdout + done.stderr


def test_source_tree_keeps_providers_out_of_the_package():
    """The source tree ships no final public ``norm/gemma_rmsnorm.py``.

    The wheel build stages that module from ``target_providers/<target>/``;
    putting a copy back into ``norm/`` would trip the staging conflict check
    instead of being silently overwritten.
    """
    assert not (NORM_ROOT / "gemma_rmsnorm.py").exists()
    assert not (NORM_ROOT / "_gemma_rmsnorm_native.py").exists()
    assert not (NORM_ROOT / "_gemma_rmsnorm_aclnn.py").exists()
    providers = PACKAGE_ROOT / "target_providers"
    assert (providers / "Ascend910" / "norm" / "gemma_rmsnorm.py").exists()
    assert (providers / "Ascend950" / "norm" / "gemma_rmsnorm.py").exists()


def test_setup_excludes_provider_tree_from_packages():
    """``target_providers/`` sits next to setup.py and holds .py files.

    Without the exclusion, find_namespace_packages would pick it up as a
    namespace package and ship the whole provider tree into the wheel.
    """
    setup_source = SETUP_PATH.read_text(encoding="utf-8")
    expected_exclude = 'exclude=("tests*", "target_providers", "target_providers.*")'
    assert expected_exclude in setup_source
