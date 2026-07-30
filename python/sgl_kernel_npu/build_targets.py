import os

from setuptools.command.build_py import build_py

BUILD_TARGET_ENV = "SGL_KERNEL_NPU_BUILD_TARGET"
DEFAULT_BUILD_TARGET = "Ascend910_9382"

_TARGET_SPECIFIC_MODULES = {
    "sgl_kernel_npu.norm.gemma_rmsnorm": frozenset({"Ascend950"}),
    "sgl_kernel_npu.norm._gemma_rmsnorm_triton": frozenset({"Ascend950"}),
}


def get_build_target() -> str:
    """Return the Ascend compilation target selected for this wheel build."""
    return os.environ.get(BUILD_TARGET_ENV, DEFAULT_BUILD_TARGET)


def module_is_enabled(module: str, build_target: str) -> bool:
    """Whether a Python kernel module belongs in the target-specific wheel."""
    supported_targets = _TARGET_SPECIFIC_MODULES.get(module)
    return supported_targets is None or build_target in supported_targets


class TargetBuildPy(build_py):
    """Exclude Python kernel modules unsupported by the wheel build target."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        build_target = get_build_target()
        return [
            module
            for module in modules
            if module_is_enabled(f"{module[0]}.{module[1]}", build_target)
        ]
