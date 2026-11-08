"""Build and package the custom ops-transformer vendor and Python bindings."""

import hashlib
import os
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
        roots = {name.split("/", 1)[0] for name in archive.namelist()}
    # CANN 9.1 predates vendor-suffixed Python packages.
    if upstream_package not in roots:
        upstream_package = "cann_ops_transformer"
    extract_python(wheel, upstream_package, destination)
    ops_root = destination / "ops"
    modules = sorted(
        ".".join(path.parent.relative_to(ops_root).parts)
        for path in ops_root.glob("*/*/__init__.py")
        if path.parent.name in item["ops"]
    )
    if not modules:
        modules = sorted(
            path.stem for path in ops_root.glob("*.py") if path.stem in item["ops"]
        )
    if not modules:
        raise RuntimeError("No selected Python bindings found in upstream wheel")
    qualified = "sgl_kernel_npu._third_party." + name
    # Upstream's builder still contains absolute imports of the unsuffixed package.
    # Rewrite Python module paths; Torch operator namespaces must remain intact.
    pattern = (
        r"(?m)^(\s*from\s+)(?:"
        + re.escape(upstream_package)
        + r"|cann_ops_transformer)(?=[.\s])"
    )
    imports = (
        r"(?m)^(\s*import\s+)("
        + re.escape(upstream_package)
        + r"|cann_ops_transformer)(\.[\w.]+)?(?:\s+as\s+(\w+))?[ \t]*$"
    )
    for path in destination.rglob("*.py"):
        source = re.sub(pattern, lambda m: m[1] + qualified, path.read_text())

        def rewrite_import(match):
            prefix, original, suffix, alias = match.groups()
            if suffix and not alias:
                return (
                    prefix
                    + qualified
                    + suffix
                    + "\n"
                    + prefix
                    + qualified
                    + " as "
                    + original
                )
            return prefix + qualified + (suffix or "") + " as " + (alias or original)

        source = re.sub(imports, rewrite_import, source)
        # CompressorV2 creates ops.ds41 using a Python module-cache lookup.
        source = source.replace(
            'sys.modules["cann_ops_transformer.ops"]',
            'sys.modules["{}.ops"]'.format(qualified),
        )
        path.write_text(source)
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
        # Template kernels retain the public name but install under *_apt.
        for installed_name in (op, op + "_apt"):
            config = kernel / "config" / item["soc"] / (installed_name + ".json")
            if not config.exists():
                continue
            read_json(require(config))
            binaries = sorted((kernel / item["soc"] / installed_name).glob("*.o"))
            if not binaries:
                raise RuntimeError("No AICore binaries for " + installed_name)
            for binary in binaries:
                require(binary)
                read_json(require(binary.with_suffix(".json")))
            covered.add(normalize(op))
        if normalize(op) not in covered:
            raise RuntimeError("No installed AICore/AICPU implementation for " + op)


def build_python_wheel(item, source, directory):
    """Use the profile command, or package CANN 9.1's selected bindings."""
    if any("--torch_extension" in command for command in item["commands"]):
        wheels = list((source / "torch_extension/dist").glob("*.whl"))
    else:
        wheel_dir = directory / "python-wheel"
        run(
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", wheel_dir],
            directory / "build-python.log",
            cwd=source / "torch_extension",
            env=dict(
                os.environ,
                TORCH_EXTENSION_OPS=",".join(item["ops"]),
                TORCH_EXTENSION_VENDOR=item["vendor_name"],
                TORCH_EXTENSION_EXPERIMENTAL="FALSE",
            ),
        )
        wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one selected upstream Python wheel")
    return wheels[0]


def build(item, source, directory):
    for index, command in enumerate(item["commands"]):
        run(command, directory / ("build-{}.log".format(index)), cwd=source)
    installers = list((source / "build_out").glob("*.run"))
    wheel = build_python_wheel(item, source, directory)
    if len(installers) != 1:
        raise RuntimeError("Expected exactly one upstream .run")
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
    return dict(
        config=item,
        vendor=vendor,
        wheel=wheel,
        license=source / item["license"],
    )


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
        item, vendor_directory=vendor.name, files=hashes, binding="upstream_jit"
    )
