"""Continue a selected CANN build through Torch and an optional binary wheel."""

from configparser import ConfigParser
import os
from pathlib import Path
import shutil
import sys
import zipfile

from build_tools.cann import PROJECT_ROOT, _remove_tree, run


def stage_runtime(vendor, package, source_root=PROJECT_ROOT):
    package.mkdir(parents=True, exist_ok=True)
    python_package = source_root / "python/sgl_kernel_npu/sgl_kernel_npu"
    shutil.copy2(python_package / "__init__.py", package / "__init__.py")
    # This is the expanded vendor, not the installer and not SDK libraries.
    shutil.copytree(vendor, package / "_cann/vendors/sgl_kernel_npu")


def build_torch(cann, arch_version, *, jobs=None, source_root=PROJECT_ROOT):
    if cann.vendor is None:
        raise RuntimeError("Torch requires a built and installed CANN package")
    source_root = Path(source_root).resolve()
    directory = cann.directory.parent / "torch"
    # Incremental compilation/recovery is a separate acceptance stage (3B).
    _remove_tree(directory)
    directory.mkdir()
    package = directory / "package/sgl_kernel_npu"
    stage_runtime(cann.vendor, package, source_root)
    (package / "_cann/selected-operators.txt").write_text("\n".join(cann.operators) + "\n")
    (package / "_cann/arch-version.txt").write_text(arch_version + "\n")
    run(["cmake", "-S", source_root / "csrc/aclnn_torch_adapter", "-B", directory,
         "-DCMAKE_BUILD_TYPE=Release",
         "-DPython3_EXECUTABLE=" + sys.executable, "-DSGL_ARCH_VERSION=" + arch_version,
         "-DSGL_CANN_OPERATORS=" + ";".join(cann.operators),
         "-DSGL_CANN_VENDOR=" + str(cann.vendor), "-DSGL_TORCH_PACKAGE=" + str(package)],
        directory / "configure.log", cwd=source_root)
    command = ["cmake", "--build", directory]
    if jobs is not None:
        command += ["--parallel", str(jobs)]
    run(command, directory / "build.log", cwd=source_root)
    for name in ("libsgl_kernel_npu.so", "libsgl_kernel_npu_compare.so"):
        path = package / "lib" / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError("Missing Torch library: {}".format(path))
    print("TORCH_PACKAGE: " + str(package), flush=True)
    print("COMPARE_LIBRARY: " + str(package / "lib/libsgl_kernel_npu_compare.so"), flush=True)
    return package


def verify_wheel(wheel, package):
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        expected = {"sgl_kernel_npu/" + p.relative_to(package).as_posix(): p
                    for p in package.rglob("*") if p.is_file()}
        for name, path in expected.items():
            if name not in names or archive.read(name) != path.read_bytes():
                raise RuntimeError("Wheel content differs from this build: " + name)
        contents = {name for name in names if name.startswith("sgl_kernel_npu/") and not name.endswith("/")}
        if contents != set(expected) or any("compare" in name for name in contents):
            raise RuntimeError("Wheel contains unexpected or comparison-only files")
        metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1 or b"Root-Is-Purelib: false" not in archive.read(metadata[0]):
            raise RuntimeError("Expected a platform-specific binary wheel")


def build_wheel(package, *, source_root=PROJECT_ROOT):
    source_root = Path(source_root).resolve()
    directory = package.parents[1].parent / "wheel"
    _remove_tree(directory)
    directory.mkdir()
    staged = directory / "package/sgl_kernel_npu"
    shutil.copytree(package, staged, ignore=shutil.ignore_patterns("libsgl_kernel_npu_compare.so", "__pycache__"))
    dist = directory / "dist"
    env = dict(os.environ, SGL_BUILD_MODE="REFACTOR", SGL_REFACTOR_STAGED_PACKAGE=str(staged))
    run([sys.executable, "setup.py", "egg_info", "--egg-base", directory,
         "build", "--build-base", directory / "setuptools",
         "bdist_wheel", "--bdist-dir", directory / "bdist", "--dist-dir", dist],
        directory / "build.log", cwd=source_root, env=env)
    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one wheel in {}".format(dist))
    verify_wheel(wheels[0], staged)
    print("WHEEL: " + str(wheels[0]), flush=True)
    print("PASS: wheel contains this build's selected CANN vendor and Torch library; NPU tests still required")
    return wheels[0]


def setup_cann():
    """Internal packaging hook, called only after the Python coordinator builds."""
    from setuptools import setup
    from setuptools.command.install import install
    from setuptools.dist import Distribution
    import torch
    import torch_npu

    value = os.environ.get("SGL_REFACTOR_STAGED_PACKAGE")
    if not value:
        raise RuntimeError("Use SGL_BUILD_MODE=REFACTOR bash build.sh --arch-version <2201|3510>")
    package = Path(value).resolve()
    if not (package / "lib/libsgl_kernel_npu.so").is_file():
        raise RuntimeError("Missing staged Torch library: {}".format(package))
    config = ConfigParser()
    config.read(PROJECT_ROOT / "config.ini")

    class BinaryDistribution(Distribution):
        def has_ext_modules(self):
            return True

    class BinaryInstall(install):
        def finalize_options(self):
            super().finalize_options()
            self.install_lib = self.install_platlib

    setup(name="sgl_kernel_npu", version=config.get("global", "version"),
          description="Selected CANN operators and Torch adapters for Ascend NPU",
          packages=["sgl_kernel_npu"], package_dir={"sgl_kernel_npu": str(package)},
          package_data={"sgl_kernel_npu": [p.relative_to(package).as_posix()
                                         for p in package.rglob("*") if p.is_file()]},
          distclass=BinaryDistribution, cmdclass={"install": BinaryInstall}, python_requires=">=3.7",
          install_requires=["torch==" + torch.__version__.split("+")[0],
                            "torch_npu==" + torch_npu.__version__.split("+")[0]],
          url="https://github.com/sgl-project/sgl-kernel-npu/", license="BSD 3 License")
