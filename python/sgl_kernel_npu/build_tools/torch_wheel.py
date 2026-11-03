"""Continue a selected CANN build through Torch and an optional binary wheel."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import zipfile
from configparser import ConfigParser
from pathlib import Path

from build_tools.cann import PROJECT_ROOT, _cache, _remove_tree, architecture, run


def stage_runtime(vendor, package, source_root=PROJECT_ROOT, arch_version="2201"):
    _remove_tree(package)
    package.parent.mkdir(parents=True, exist_ok=True)
    python_package = source_root / "python/sgl_kernel_npu/sgl_kernel_npu"
    shutil.copytree(
        python_package,
        package,
        ignore=shutil.ignore_patterns(
            "lib", "_cann", "__pycache__", "*.pyc", "*.so", "*.egg-info"
        ),
    )
    from build_tools.target_provider import stage_target_providers

    stage_target_providers(
        source_root / "python/sgl_kernel_npu",
        package.parent,
        architecture(arch_version, source_root)["provider"],
    )
    shutil.copy2(source_root / "config.ini", package / "config.ini")
    # This is the expanded vendor, not the installer and not SDK libraries.
    shutil.copytree(vendor, package / "_cann/vendors/sgl_kernel_npu")


def build_torch(cann, arch_version, *, jobs=None, source_root=PROJECT_ROOT):
    if cann.vendor is None:
        raise RuntimeError("Torch requires a built and installed CANN package")
    source_root = Path(source_root).resolve()
    directory = cann.directory.parent / "torch"
    if directory.is_symlink():
        raise RuntimeError("Refusing a symlinked Torch build directory")
    expected = {
        "CMAKE_HOME_DIRECTORY": str(source_root / "csrc/aclnn_torch_adapter"),
        "SGL_ARCH_VERSION": arch_version,
        "SGL_CANN_OPERATORS": ";".join(cann.operators),
        "Python3_EXECUTABLE": sys.executable,
        "SGL_TORCH_LIBRARY_DIR": str(directory / "lib"),
    }
    cached = _cache(directory)
    if cached and any(cached.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Torch build configuration changed; rerun with --make_clean")
    environment = {
        "python": sys.executable,
        "python_version": sys.version,
        "sdk": os.environ.get("ASCEND_HOME_PATH", ""),
    }
    for distribution in ("torch", "torch_npu"):
        try:
            environment[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            environment[distribution] = None  # CMake validates real Torch imports.
    signature = directory / "torch-environment.json"
    if signature.exists() and json.loads(signature.read_text()) != environment:
        raise RuntimeError(
            "Torch/Python/SDK environment changed; rerun with --make_clean"
        )
    directory.mkdir(parents=True, exist_ok=True)
    signature.write_text(json.dumps(environment, sort_keys=True) + "\n")
    package = directory / "package/sgl_kernel_npu"
    stage_runtime(cann.vendor, package, source_root, arch_version)
    (package / "_cann/selected-operators.txt").write_text(
        "\n".join(cann.operators) + "\n"
    )
    (package / "_cann/arch-version.txt").write_text(arch_version + "\n")
    manifest = {
        "schema_version": 1,
        "build_mode": "REFACTOR",
        "arch_version": arch_version,
        "selection_kind": cann.selection_kind,
        "operators": sorted(cann.operators),
        "compute_units": architecture(arch_version, source_root)["compute_units"],
    }
    (package / "_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run(
        [
            "cmake",
            "-S",
            source_root / "csrc/aclnn_torch_adapter",
            "-B",
            directory,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DSGL_ARCH_VERSION=" + arch_version,
            "-DSGL_CANN_OPERATORS=" + ";".join(cann.operators),
            "-DSGL_CANN_VENDOR=" + str(cann.vendor),
            "-DSGL_TORCH_LIBRARY_DIR=" + str(directory / "lib"),
        ],
        directory / "configure.log",
        cwd=source_root,
    )
    command = ["cmake", "--build", directory]
    if jobs is not None:
        command += ["--parallel", str(jobs)]
    run(command, directory / "build.log", cwd=source_root)
    library = directory / "lib/libsgl_kernel_npu.so"
    if not library.is_file() or not library.stat().st_size:
        raise RuntimeError("Missing Torch library: {}".format(library))
    (package / "lib").mkdir()
    shutil.copy2(library, package / "lib" / library.name)
    print("TORCH_PACKAGE: " + str(package), flush=True)
    return package


def wheel_build_tag(manifest):
    kind = manifest["selection_kind"]
    if kind not in ("all", "custom"):
        raise ValueError("Invalid wheel selection kind")
    suffix = ""
    if kind == "custom":
        suffix = hashlib.sha256(
            ",".join(sorted(manifest["operators"])).encode()
        ).hexdigest()[:10]
    return "1refactor" + manifest["arch_version"] + kind + suffix


def verify_wheel(wheel, package):
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        expected = {
            "sgl_kernel_npu/" + p.relative_to(package).as_posix(): p
            for p in package.rglob("*")
            if p.is_file()
        }
        for name, path in expected.items():
            if name not in names or archive.read(name) != path.read_bytes():
                raise RuntimeError("Wheel content differs from this build: " + name)
        contents = {
            name
            for name in names
            if name.startswith("sgl_kernel_npu/") and not name.endswith("/")
        }
        if (
            contents != set(expected)
            or "sgl_kernel_npu/lib/libsgl_kernel_npu_compare.so" in contents
        ):
            raise RuntimeError("Wheel contains unexpected or comparison-only files")
        manifest = json.loads(archive.read("sgl_kernel_npu/_build_manifest.json"))
        if "-" + wheel_build_tag(manifest) + "-" not in wheel.name:
            raise RuntimeError("Wheel build tag does not match its operator manifest")
        metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1 or b"Root-Is-Purelib: false" not in archive.read(
            metadata[0]
        ):
            raise RuntimeError("Expected a platform-specific binary wheel")


def build_wheel(package, *, source_root=PROJECT_ROOT):
    source_root = Path(source_root).resolve()
    directory = package.parents[1].parent / "wheel"
    _remove_tree(directory)
    directory.mkdir()
    staged = directory / "package/sgl_kernel_npu"
    shutil.copytree(
        package,
        staged,
        ignore=shutil.ignore_patterns("libsgl_kernel_npu_compare.so", "__pycache__"),
    )
    dist = directory / "dist"
    manifest = json.loads((staged / "_build_manifest.json").read_text())
    env = dict(os.environ, SGL_REFACTOR_STAGED_PACKAGE=str(staged))
    run(
        [
            sys.executable,
            "setup.py",
            "egg_info",
            "--egg-base",
            directory,
            "build",
            "--build-base",
            directory / "setuptools",
            "bdist_wheel",
            "--build-number",
            wheel_build_tag(manifest),
            "--bdist-dir",
            directory / "bdist",
            "--dist-dir",
            dist,
        ],
        directory / "build.log",
        cwd=source_root,
        env=env,
    )
    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one wheel in {}".format(dist))
    verify_wheel(wheels[0], staged)
    print("WHEEL: " + str(wheels[0]), flush=True)
    print(
        "PASS: wheel contains this build's selected CANN vendor and Torch library; NPU tests still required"
    )
    return wheels[0]


def setup_refactor():
    """Internal packaging hook, called only after the Python coordinator builds."""
    import torch
    import torch_npu
    from setuptools import find_namespace_packages, setup
    from setuptools.command.install import install
    from setuptools.dist import Distribution

    value = os.environ.get("SGL_REFACTOR_STAGED_PACKAGE")
    if not value:
        raise RuntimeError(
            "Use the repository-root setup.py --arch-version <2201|3510>"
        )
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

    setup(
        name="sgl_kernel_npu",
        version=config.get("global", "version"),
        description="Selected CANN operators and Torch adapters for Ascend NPU",
        packages=find_namespace_packages(
            where=str(package.parent),
            include=["sgl_kernel_npu", "sgl_kernel_npu.*"],
            exclude=["sgl_kernel_npu.lib*", "sgl_kernel_npu._cann*"],
        ),
        package_dir={"": str(package.parent)},
        package_data={
            "sgl_kernel_npu": [
                p.relative_to(package).as_posix()
                for p in package.rglob("*")
                if p.is_file()
            ]
        },
        distclass=BinaryDistribution,
        cmdclass={"install": BinaryInstall},
        python_requires=">=3.7",
        install_requires=[
            "torch==" + torch.__version__.split("+")[0],
            "torch_npu==" + torch_npu.__version__.split("+")[0],
        ],
        url="https://github.com/sgl-project/sgl-kernel-npu/",
        license="BSD 3 License",
    )
