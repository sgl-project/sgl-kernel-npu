"""Build selected recipes kernels and isolated, precompiled Torch bindings."""

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

from build_tools.cann import run
from build_tools.third_party import detect_soc_version, extract_python, read_json


def configure(config, jobs):
    config = dict(config)
    operators = config["ops"]
    if not isinstance(operators, dict) or not operators:
        raise ValueError("recipes ops must map Python interface names to kernel names")
    for interface, kernel in operators.items():
        if any(
            not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            for name in (interface, kernel)
        ):
            raise ValueError("Invalid recipes interface or kernel name")
    config["soc"] = detect_soc_version()
    supported = {"2201": ("ascend910b", "ascend910_93"), "3510": ("ascend950",)}
    if config["soc"] not in supported[config["arch_version"]]:
        raise RuntimeError("Detected recipes SoC does not match selected architecture")
    if config.get("has_aicpu", False):
        raise ValueError("Selected recipes operators do not contain AICPU kernels")
    config["vendor_name"] = "vendor_" + config["name"]
    config["python_package"] = config["name"]
    config["jobs"] = jobs or 8
    config["kernel_ops"] = ";".join(dict.fromkeys(operators.values()))
    values = dict(config, python=sys.executable)
    config["commands"] = [
        [argument.format(**values) for argument in command]
        for command in config["commands"]
    ]
    return config


def prepare_python(source, item):
    """Keep upstream kernels/bindings, selecting schemas and isolating namespace."""
    extension = source / "ops/ascendc/torch_ops_extension"
    original = extension / "custom_ops"
    destination = extension / item["python_package"]
    destination.mkdir()
    sources = destination / "csrc"
    sources.mkdir()
    upstream = original / "csrc"
    for filename in ("ops_common.h", "ops_common.cpp"):
        shutil.copy2(upstream / filename, sources / filename)
    declarations = (upstream / "ops_def_registration.cpp").read_text()
    schemas = []
    for op in item["ops"]:
        matches = re.findall(r'm\.def\("' + re.escape(op) + r"\([^;]+;", declarations)
        if len(matches) != 1:
            raise RuntimeError("Expected one upstream schema for " + op)
        schemas.append(matches[0])
        content = (upstream / (op + ".cpp")).read_text()
        content = content.replace(
            "TORCH_LIBRARY_IMPL(custom,", "TORCH_LIBRARY_IMPL(" + item["name"] + ","
        )
        (sources / (op + ".cpp")).write_text(content)
    notice = declarations[: declarations.index("#include")]
    (sources / "ops_def_registration.cpp").write_text(
        notice
        + "#include <torch/extension.h>\n#include <torch/library.h>\n"
        + "TORCH_LIBRARY("
        + item["name"]
        + ", m) {\n"
        + "\n".join(schemas)
        + "\n}\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}\n"
    )
    (destination / "__init__.py").write_text(
        "import torch\nimport torch_npu\nfrom . import custom_ops_lib\n"
        + "\n".join(
            "{} = torch.ops.{}.{}".format(op, item["name"], op) for op in item["ops"]
        )
        + "\n"
    )
    # Build the selected upstream C++ implementations; no installation or JIT.
    (extension / "sgl_setup.py").write_text(
        "import os\nfrom pathlib import Path\nimport torch_npu\n"
        "from setuptools import setup\n"
        "from torch.utils.cpp_extension import BuildExtension\n"
        "from torch_npu.utils.cpp_extension import NpuExtension\n"
        "base = Path(__file__).resolve().parent\nos.chdir(base)\n"
        "name = " + repr(item["python_package"]) + "\n"
        "setup(name=name, version='1.0.0', packages=[name],\n"
        "    ext_modules=[NpuExtension(name + '.custom_ops_lib',\n"
        "        sources=[str(p) for p in (base / name / 'csrc').glob('*.cpp')],\n"
        "        extra_compile_args=['-O2', '-fvisibility=hidden',\n"
        "            '-I' + str(Path(torch_npu.__file__).parent / 'include/third_party/acl/inc')])],\n"
        "    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)})\n"
    )


def inspect_vendor(vendor, item):
    vendor = Path(vendor)

    def require(path):
        if (
            not path.is_file()
            or not path.stat().st_size
            or vendor.resolve() not in path.resolve().parents
        ):
            raise RuntimeError("Missing or external recipes resource: " + str(path))
        return path

    require(vendor / "op_api/lib/libcust_opapi.so")
    kernel = vendor / "op_impl/ai_core/tbe/kernel"
    for op in item["ops"]:
        name = item["ops"][op]
        read_json(require(kernel / "config" / item["soc"] / (name + ".json")))
        binaries = sorted((kernel / item["soc"] / name).glob("*.o"))
        if not binaries:
            raise RuntimeError("No recipes AICore binaries for " + op)
        for binary in binaries:
            require(binary)
            read_json(require(binary.with_suffix(".json")))


def build(item, source, directory):
    prepare_python(source, item)
    env = dict(os.environ, MAX_JOBS=str(item["jobs"]))
    for index, command in enumerate(item["commands"]):
        run(command, directory / "build-{}.log".format(index), cwd=source, env=env)
    installers = list((source / "ops/ascendc/build").glob("*.run"))
    wheels = list((source / "ops/ascendc/torch_ops_extension/dist").glob("*.whl"))
    if len(installers) != 1 or len(wheels) != 1:
        raise RuntimeError("Expected one recipes .run and one selected Python wheel")
    installed = directory / "installed"
    installed.mkdir()
    run(
        ["bash", installers[0], "--quiet", "--install-path=" + str(installed)],
        directory / "install.log",
        cwd=directory,
    )
    vendor = installed / "vendors" / item["vendor_name"]
    inspect_vendor(vendor, item)
    return dict(
        config=item, vendor=vendor, wheel=wheels[0], license=source / item["license"]
    )


def stage(result, package):
    item = result["config"]
    vendor = result["vendor"]
    for path in vendor.rglob("*"):
        if path.is_symlink() and vendor.resolve() not in path.resolve().parents:
            raise RuntimeError("External recipes vendor symlink: " + str(path))
    target = package / "_cann/vendors" / vendor.name
    if target.exists():
        raise RuntimeError("Duplicate recipes vendor: " + str(target))
    shutil.copytree(vendor, target)
    if (target / "bin").exists():
        shutil.rmtree(target / "bin")
    extract_python(
        result["wheel"],
        item["python_package"],
        package / "_third_party" / item["python_package"],
    )
    licenses = package / "_third_party/licenses" / item["name"]
    licenses.mkdir(parents=True)
    shutil.copy2(result["license"], licenses / "LICENSE")
    files = {
        path.relative_to(target)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in target.rglob("*")
        if path.is_file()
    }
    return dict(item, vendor_directory=vendor.name, files=files, binding="precompiled")
