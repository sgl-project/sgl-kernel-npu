from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "python" / "sgl_kernel_npu"


def test_python_build_has_no_generated_target_config():
    setup_source = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "TargetBuildPy" not in setup_source
    assert "SGL_KERNEL_NPU_BUILD_TARGET" not in setup_source
    assert not (PACKAGE_ROOT / "sgl_kernel_npu" / "_build_target.py").exists()


def test_build_script_passes_compiled_gemma_capability():
    source = (PROJECT_ROOT / "build.sh").read_text(encoding="utf-8")

    assert 'SGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM="ON"' in source
    assert 'SGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM="OFF"' in source
    assert (
        '"-DSGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM='
        '$SGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM"' in source
    )
    assert "SGL_KERNEL_NPU_BUILD_TARGET" not in source


def test_cmake_compiles_gemma_capability_into_host_library():
    root_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    kernel_cmake = (PROJECT_ROOT / "csrc" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "SGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM" in root_cmake
    assert "target_compile_definitions(" in kernel_cmake
    assert "SGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM" in kernel_cmake


def test_host_library_exposes_compiled_gemma_capability():
    source = (PROJECT_ROOT / "csrc" / "pytorch_extensions.cpp").read_text(
        encoding="utf-8"
    )

    assert "sgl_kernel_npu_use_native_gemma_rms_norm() -> bool" in source
    assert "#ifdef SGL_KERNEL_NPU_USE_NATIVE_GEMMA_RMS_NORM" in source


def test_gemma_module_uses_no_runtime_soc_dispatch():
    source = (PACKAGE_ROOT / "sgl_kernel_npu" / "norm" / "gemma_rmsnorm.py").read_text(
        encoding="utf-8"
    )

    assert "sgl_kernel_npu_use_native_gemma_rms_norm()" in source
    assert "_build_target" not in source
    assert "get_soc_version" not in source
    assert "NpuDeviceFamily" not in source
