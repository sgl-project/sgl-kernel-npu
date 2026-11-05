"""Build and stage the opt-in, namespaced Ascend 950 ZeRCMoE dependency."""

import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from build_tools.cann import PROJECT_ROOT, run
from build_tools.local_inputs import local_input_hashes
from build_tools.third_party import detect_soc_version, read_json

_SOURCE_ROOT = PROJECT_ROOT / "csrc/zerc_moe"
_BINDINGS = (
    "op_builder/builder.py",
    "ops/comm_context.py",
    "ops/mega_moe.py",
    "ops/csrc/comm_context.cpp",
    "ops/csrc/mega_moe.cpp",
    "ops/graph_convert/graph_convert_mega_moe.py",
    "common/inc/aclnn_common.h",
    "common/inc/hccl_common.h",
)


def _overlay_digest():
    hashes = local_input_hashes(
        (
            "csrc/zerc_moe/source",
            "csrc/zerc_moe/build.py",
            "csrc/zerc_moe/LICENSE",
            "python/sgl_kernel_npu/build_tools/local_inputs.py",
        )
    )
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def configure(config, jobs):
    """Resolve an explicit arch35-only build and fingerprint local overlays."""
    config = dict(config)
    if config["arch_version"] != "3510":
        raise ValueError("ZeRCMoE requires arch35")
    if detect_soc_version() != "ascend950":
        raise RuntimeError("ZeRCMoE requires an Ascend 950 build host")
    config["soc"] = "ascend950"
    config["vendor_name"] = "vendor_" + config["name"]
    config["python_package"] = config["name"]
    config["jobs"] = jobs or 8
    config["commands"] = [
        [
            "bash",
            "build.sh",
            "--pkg",
            "--soc=ascend950",
            "--vendor_name=" + config["vendor_name"],
            "--ops=zerc_moe",
            "-j" + str(config["jobs"]),
        ],
        ["python3", "-m", "build", "--wheel", "--no-isolation"],
    ]
    config["overlay_sha256"] = _overlay_digest()
    return config


def prepare_source(source):
    """Overlay checked-in files and select the separate MC2 operator name."""
    source = Path(source)
    spec = importlib.util.spec_from_file_location(
        "_zerc_moe_source_overlay", _SOURCE_ROOT / "build.py"
    )
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    overlay.overlay_sources(source)

    original = source / "mc2/mega_moe"
    isolated = source / "mc2/zerc_moe"
    if isolated.exists() or isolated.is_symlink():
        raise RuntimeError("Namespaced ZeRCMoE already exists in pinned framework")
    shutil.copytree(original, isolated, symlinks=False)
    for old, new in (
        ("op_host/mega_moe_def.cpp", "op_host/zerc_moe_def.cpp"),
        ("op_kernel/mega_moe.cpp", "op_kernel/zerc_moe.cpp"),
        ("op_api/aclnn_mega_moe.cpp", "op_api/aclnn_zerc_moe.cpp"),
    ):
        old_path, new_path = isolated / old, isolated / new
        if not old_path.is_file() or new_path.exists():
            raise RuntimeError("Unexpected operator source layout: " + str(old_path))
        old_path.rename(new_path)
    cmake = source / "cmake/func.cmake"
    text = cmake.read_text()
    marker = '        "mega_moe;"'
    if text.count(marker) != 1:
        raise RuntimeError("Pinned MC2 operator list does not match expectations")
    cmake.write_text(text.replace(marker, marker + '\n        "zerc_moe;"'))
    builder = source / "torch_extension/npu_ops_transformer/op_builder/builder.py"
    text = builder.read_text()
    old_library = 'Library("npu_ops_transformer", "DEF")'
    if text.count(old_library) != 1:
        raise RuntimeError("Pinned Python builder layout does not match expectations")
    builder.write_text(text.replace(old_library, 'Library("zerc_moe", "DEF")'))
    ops_init = source / "torch_extension/npu_ops_transformer/ops/__init__.py"
    text = ops_init.read_text()
    header, marker, _ = text.partition("import os as _os\n")
    if not marker or "NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE" not in text:
        raise RuntimeError("Pinned Python ops initializer does not match expectations")
    # The standalone wheel must not eagerly import unrelated upstream ops using
    # the old Torch namespace. Preserve its original license notice.
    ops_init.write_text(
        header + '"""ZeRCMoE bindings; import the required module explicitly."""\n'
    )


def inspect_vendor(vendor, item):
    """Check that the selected Ascend 950 kernel and op API are packaged."""
    vendor = Path(vendor)

    def require(path):
        if (
            not path.is_file()
            or not path.stat().st_size
            or vendor.resolve() not in path.resolve().parents
        ):
            raise RuntimeError("Missing or external vendor resource: " + str(path))
        return path

    require(vendor / "op_api/lib/libcust_opapi.so")
    for pattern in (
        "op_proto/lib/linux/*/libcust_opsproto_rt2.0.so",
        "op_impl/ai_core/tbe/op_tiling/lib/linux/*/libcust_opmaster_rt2.0.so",
    ):
        libraries = list(vendor.glob(pattern))
        if len(libraries) != 1:
            raise RuntimeError("Expected one installed MC2 library: " + pattern)
        require(libraries[0])
    kernel = vendor / "op_impl/ai_core/tbe/kernel"
    config = kernel / "config" / item["soc"] / "zerc_moe.json"
    read_json(require(config))
    binaries = sorted((kernel / item["soc"] / "zerc_moe").glob("*.o"))
    if not binaries:
        raise RuntimeError("No ZeRCMoE AICore binaries")
    for binary in binaries:
        require(binary)
        read_json(require(binary.with_suffix(".json")))


def build(item, source, directory):
    missing = []
    for distribution in ("build", "setuptools", "wheel"):
        try:
            importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError("Install Python build dependencies: " + ", ".join(missing))
    if _overlay_digest() != item["overlay_sha256"]:
        raise RuntimeError("ZeRCMoE source changed since the build was planned")
    prepare_source(source)
    if _overlay_digest() != item["overlay_sha256"]:
        raise RuntimeError("ZeRCMoE source changed while preparing the build")
    run(
        item["commands"][0],
        directory / "build-0.log",
        cwd=source,
    )
    run(
        [sys.executable, *item["commands"][1][1:]],
        directory / "build-1.log",
        cwd=source / "torch_extension",
    )
    installers = list((source / "build_out").glob("*.run"))
    wheels = list((source / "torch_extension/dist").glob("*.whl"))
    if len(installers) != 1 or len(wheels) != 1:
        raise RuntimeError("Expected one installer and one Python wheel")
    installed = directory / "installed"
    installed.mkdir()
    run(
        ["bash", installers[0], "--quiet", "--install-path=" + str(installed)],
        directory / "install.log",
        cwd=directory,
    )
    vendors = list((installed / "vendors").glob("*"))
    if len(vendors) != 1 or not vendors[0].is_dir() or vendors[0].is_symlink():
        raise RuntimeError("Expected one installed vendor directory")
    inspect_vendor(vendors[0], item)
    return dict(
        config=item, vendor=vendors[0], wheel=wheels[0], license=source / "LICENSE"
    )


def stage_python(wheel, package):
    """Extract only ZeRCMoE bindings and give the JIT builder a private namespace."""
    destination = package / "_third_party/zerc_moe"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe wheel member: " + name)
        for relative in _BINDINGS:
            member = "npu_ops_transformer/" + relative
            if names.count(member) != 1:
                raise RuntimeError("Missing or duplicate ZeRCMoE binding: " + member)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    for relative in (
        "__init__.py",
        "op_builder/__init__.py",
        "ops/__init__.py",
        "ops/graph_convert/__init__.py",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('"""ZeRCMoE bundled bindings."""\n')
    builder = destination / "op_builder/builder.py"
    text = builder.read_text()
    old_import = "import npu_ops_transformer"
    private_library = 'Library("zerc_moe", "DEF")'
    if text.count(old_import) != 1 or text.count(private_library) != 1:
        raise RuntimeError("Pinned Python builder layout does not match expectations")
    text = text.replace(
        old_import,
        "from sgl_kernel_npu._third_party import zerc_moe as npu_ops_transformer",
    )
    builder.write_text(text)
    for relative in ("ops/mega_moe.py", "ops/comm_context.py"):
        path = destination / relative
        text = path.read_text().replace(
            "from npu_ops_transformer.op_builder.builder import",
            "from ..op_builder.builder import",
        )
        if "npu_ops_transformer.op_builder" in text:
            raise RuntimeError("Unisolated ZeRCMoE binding imports: " + str(path))
        path.write_text(text)


def stage(result, package):
    item = result["config"]
    vendor = Path(result["vendor"])
    for path in vendor.rglob("*"):
        if path.is_symlink() and vendor.resolve() not in path.resolve().parents:
            raise RuntimeError("External vendor symlink: " + str(path))
    target = package / "_cann/vendors" / vendor.name
    if target.exists():
        raise RuntimeError("Duplicate vendor: " + str(target))
    shutil.copytree(vendor, target)
    if (target / "bin").exists():
        shutil.rmtree(target / "bin")
    stage_python(result["wheel"], package)
    licenses = package / "_third_party/licenses" / item["name"]
    licenses.mkdir(parents=True)
    shutil.copy2(result["license"], licenses / "LICENSE")
    hashes = {
        p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    return dict(
        item, vendor_directory=vendor.name, files=hashes, binding="upstream_jit"
    )
