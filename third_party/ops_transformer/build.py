"""Build and package the custom ops-transformer vendor and Python bindings."""

import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

from build_tools.cann import run
from build_tools.third_party import detect_soc_version, extract_python, read_json


def configure(config, jobs):
    """Resolve ops-transformer's names and command placeholders."""
    config = dict(config)
    config["soc"] = detect_soc_version()
    supported_socs = {
        "2201": ("ascend910b", "ascend910_93"),
        "3510": ("ascend950",),
    }
    if config["soc"] not in supported_socs[config["arch_version"]]:
        raise RuntimeError(
            "Detected SoC {} does not match architecture {}".format(
                config["soc"], config["arch_version"]
            )
        )
    config["vendor_name"] = "vendor_" + config["name"]
    config["python_package"] = config["name"]
    config["python_bindings"] = config.get("python_bindings", True)
    if not isinstance(config["python_bindings"], bool):
        raise ValueError("python_bindings must be a boolean")
    config["source_patches"] = {}
    if config["soc"] == "ascend950":
        # The 9.2 SDK cannot generate APIs for the 960-only HiFloat4 schemas.
        for operator, filename in (
            ("indexer_quant_cache", "ascend950-indexer-quant-cache.patch"),
            ("quant_flash_attn", "ascend950-quant-flash-attn.patch"),
        ):
            if operator in config["ops"]:
                patch = Path(__file__).parent / "patches" / filename
                config["source_patches"][patch.name] = hashlib.sha256(
                    patch.read_bytes()
                ).hexdigest()
    values = dict(config, jobs=jobs or 8)
    values["ops"] = ",".join(config["ops"])
    config["commands"] = [
        [arg.format(**values) for arg in command] for command in config["commands"]
    ]
    return config


def stage_python(wheel, item, package):
    name = item["name"]
    destination = package / "_third_party" / item["python_package"]
    # Upstream adds its own package prefix; only its temporary wheel uses it.
    upstream_package = "cann_ops_transformer_" + item["vendor_name"]
    with zipfile.ZipFile(wheel) as archive:
        if "cann_ops_transformer/__init__.py" in archive.namelist():
            upstream_package = "cann_ops_transformer"
    extract_python(wheel, upstream_package, destination)
    ops_root = destination / "ops"
    modules = sorted(
        ".".join(path.parent.relative_to(ops_root).parts)
        for path in ops_root.glob("*/*/__init__.py")
        if path.parent.name in item["ops"]
    )
    if not modules:
        modules = [op for op in item["ops"] if (ops_root / (op + ".py")).is_file()]
    if not modules:
        raise RuntimeError("No selected Python bindings found in upstream wheel")
    qualified = "sgl_kernel_npu._third_party." + name
    # Upstream's builder still contains absolute imports of the unsuffixed package.
    # Rewrite imports only; Torch operator namespaces are not Python module paths.
    pattern = (
        r"(?m)^(\s*(?:from|import)\s+)(?:"
        + re.escape(upstream_package)
        + r"|cann_ops_transformer)(?=[.\s])"
    )
    for path in destination.rglob("*.py"):
        text = re.sub(
            r"(?m)^([ \t]*import[ \t]+)cann_ops_transformer[ \t]*$",
            lambda m: m[1] + qualified + " as cann_ops_transformer",
            path.read_text(),
        )
        path.write_text(re.sub(pattern, lambda m: m[1] + qualified, text))
    # Do not discover globally installed entry points or import unrelated operators.
    (destination / "__init__.py").write_text("from .ops import *  # noqa: F403\n")
    (destination / "ops/__init__.py").write_text(
        "".join(
            "from .{} import *  # noqa: F403\n".format(module) for module in modules
        )
    )


def inspect_vendor(vendor, item):
    """Validate the custom ops-transformer layout, including selected op coverage."""
    vendor = Path(vendor)

    def require(path):
        if (
            not path.is_file()
            or not path.stat().st_size
            or vendor.resolve() not in path.resolve().parents
        ):
            raise RuntimeError("Missing or external vendor resource: " + str(path))
        return path

    def normalize(name):
        return name.replace("_", "").lower()

    require(vendor / "op_api/lib/libcust_opapi.so")
    covered = set()
    cpu_configs = sorted((vendor / "op_impl/cpu/config").glob("*.json"))
    cpu_count = 0
    for path in cpu_configs:
        declarations = read_json(require(path))
        for name, declaration in declarations.items():
            info = declaration.get("opInfo", {})
            library = info.get("kernelSo")
            if not library:
                raise RuntimeError("Missing AICPU kernelSo in " + str(path))
            if Path(library).name != library:
                raise RuntimeError("Invalid AICPU library name: " + library)
            matches = list((vendor / "op_impl/cpu").rglob(library))
            if len(matches) != 1:
                raise RuntimeError("Expected one AICPU library: " + library)
            require(matches[0])
            covered.add(normalize(name))
            cpu_count += 1
    if item.get("has_aicpu", False) != bool(cpu_count):
        raise RuntimeError("has_aicpu does not match installed AICPU declarations")

    kernel = vendor / "op_impl/ai_core/tbe/kernel"
    for op in item["ops"]:
        config = kernel / "config" / item["soc"] / (op + ".json")
        if config.exists():
            read_json(require(config))
            binaries = sorted((kernel / item["soc"] / op).glob("*.o"))
            if not binaries:
                raise RuntimeError("No AICore binaries for " + op)
            for binary in binaries:
                require(binary)
                read_json(require(binary.with_suffix(".json")))
            covered.add(normalize(op))
        if normalize(op) not in covered:
            raise RuntimeError("No installed AICore/AICPU implementation for " + op)


def prepare_source(item, source, directory):
    """Apply checked target-specific patches before compiling upstream code."""
    for name, expected in item["source_patches"].items():
        patch = Path(__file__).parent / "patches" / name
        if hashlib.sha256(patch.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Third-party source patch changed: " + name)
        run(
            ["git", "apply", "--check", patch],
            directory / (name + ".check.log"),
            cwd=source,
        )
        run(
            ["git", "apply", patch],
            directory / (name + ".apply.log"),
            cwd=source,
        )


def build(item, source, directory):
    prepare_source(item, source, directory)
    for index, command in enumerate(item["commands"]):
        run(command, directory / ("build-{}.log".format(index)), cwd=source)
    extension = source / "torch_extension"
    # CANN 9.1 has a flat Python package and a setuptools entry point, rather
    # than the per-operator --torch_extension builder shipped in CANN 9.2.
    if item["python_bindings"] and any(
        (extension / "cann_ops_transformer/ops" / (op + ".py")).is_file()
        for op in item["ops"]
    ):
        run(
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", "dist"],
            directory / "build-python.log",
            cwd=extension,
        )
    installers = list((source / "build_out").glob("*.run"))
    wheels = list((source / "torch_extension/dist").glob("*.whl"))
    if len(installers) != 1 or (item["python_bindings"] and len(wheels) != 1):
        raise RuntimeError("Expected one upstream .run and the selected Python wheel")
    installed = directory / "installed"
    installed.mkdir()
    run(
        ["bash", installers[0], "--quiet", "--install-path=" + str(installed)],
        directory / "install.log",
        cwd=directory,
    )
    vendors = [path for path in (installed / "vendors").glob("*") if path.is_dir()]
    if len(vendors) != 1 or vendors[0].is_symlink():
        raise RuntimeError("Expected exactly one installed vendor directory")
    vendor = vendors[0]
    inspect_vendor(vendor, item)
    result = dict(config=item, vendor=vendor, license=source / item["license"])
    if item["python_bindings"]:
        result["wheel"] = wheels[0]
    return result


def stage(result, package):
    item = result["config"]
    vendor = result["vendor"]
    for path in vendor.rglob("*"):
        if path.is_symlink() and vendor.resolve() not in path.resolve().parents:
            raise RuntimeError("External vendor symlink: " + str(path))
    target = package / "_cann/vendors" / vendor.name
    if target.exists():
        raise RuntimeError("Duplicate vendor: " + str(target))
    shutil.copytree(vendor, target)
    # Installer-generated scripts embed the temporary absolute installation path.
    if (target / "bin").exists():
        shutil.rmtree(target / "bin")
    if item["python_bindings"]:
        stage_python(result["wheel"], item, package)
    licenses = package / "_third_party/licenses" / item["name"]
    licenses.mkdir(parents=True)
    shutil.copy2(result["license"], licenses / "LICENSE")
    hashes = {
        p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    return dict(
        item,
        vendor_directory=vendor.name,
        files=hashes,
        binding="upstream_jit" if item["python_bindings"] else "vendor_only",
    )
