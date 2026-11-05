"""Stage native build results, create binary wheels and verify their contents."""

import hashlib
import json
import os
import shutil
import sys
import zipfile
from configparser import ConfigParser
from pathlib import Path

from build_tools.common import PROJECT_ROOT, _remove_tree, architecture, run


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
    if vendor is not None:
        shutil.copytree(vendor, package / "_cann/vendors/sgl_kernel_npu")


def stage_package(torch, *, cann=None, source_root=PROJECT_ROOT, dependencies=()):
    """Stage already-built native artifacts and Python sources; never compile."""
    source_root = Path(source_root).resolve()
    arch_version = torch.arch_version
    cann_ids = cann.operators if cann is not None else ()
    host_ids = torch.operators
    vendor = cann.vendor if cann is not None else None
    if (
        vendor != torch.cann_vendor
        or cann_ids != torch.cann_operators
        or (
            cann is not None
            and (
                cann.arch_version != arch_version
                or cann.selection_kind != torch.selection_kind
                or vendor is None
            )
        )
    ):
        raise ValueError("CANN payload differs from the built Torch extension")
    if (
        torch.library is None
        or not torch.library.is_file()
        or not torch.library.stat().st_size
    ):
        raise RuntimeError("Missing Torch library: " + str(torch.library))
    directory = torch.directory
    package = directory / "package/sgl_kernel_npu"
    stage_runtime(vendor, package, source_root, arch_version)
    from build_tools.third_party import stage_dependencies

    third_party = stage_dependencies(dependencies, package)
    if cann_ids:
        (package / "_cann/selected-operators.txt").write_text(
            "\n".join(cann_ids) + "\n"
        )
        (package / "_cann/arch-version.txt").write_text(arch_version + "\n")
    manifest = {
        "schema_version": 2,
        "build_mode": "REFACTOR",
        "arch_version": arch_version,
        "selection_kind": torch.selection_kind,
        # CANN dependency expansion may add IDs; record what was actually built.
        "operators": sorted((*cann_ids, *host_ids)),
        "operators_by_type": {"cann": sorted(cann_ids), "cpp": sorted(host_ids)},
        "third_party": third_party,
        "compute_units": (
            architecture(arch_version, source_root)["compute_units"] if cann_ids else []
        ),
    }
    (package / "_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (package / "lib").mkdir()
    shutil.copy2(torch.library, package / "lib" / torch.library.name)
    print("TORCH_PACKAGE: " + str(package), flush=True)
    return package


def wheel_build_tag(manifest):
    kind = manifest["selection_kind"]
    if kind not in ("all", "custom"):
        raise ValueError("Invalid wheel selection kind")
    # Preserve main's tag format and third-party fingerprint. Native ALL is also
    # hashed because native selections can contain CANN, CPP or both.
    parts = ["1", "refactor", manifest["arch_version"], kind]
    if kind == "custom" or manifest.get("schema_version", 1) >= 2:
        parts.append(
            hashlib.sha256(
                ",".join(sorted(manifest["operators"])).encode()
            ).hexdigest()[:10]
        )
    if manifest.get("third_party"):
        parts.extend(
            [
                "tp",
                hashlib.sha256(
                    json.dumps(manifest["third_party"], sort_keys=True).encode()
                ).hexdigest()[:10],
            ]
        )
    return "_".join(parts)


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
        "PASS: wheel matches this build's native operator manifest and Torch library; NPU tests still required"
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
        description="Selected CANN and C++ operators for Ascend NPU",
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
