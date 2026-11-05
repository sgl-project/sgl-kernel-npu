"""Bundle selected pinned native operators without installed operator plugins."""

import hashlib
import re
import shutil
import sys
from pathlib import Path

from build_tools.cann import run
from build_tools.third_party import detect_soc_version, extract_python

DISPATCH_KEYS = {
    "compressor": "PrivateUse1",
    "sparse_attn_sharedkv": "PrivateUse1",
    "sparse_attn_sharedkv_metadata_host": "CPU",
    "kv_quant_sparse_attn_sharedkv": "PrivateUse1",
    "kv_quant_sparse_attn_sharedkv_metadata_host": "CatchAll",
    "kv_compress_epilog": "PrivateUse1",
}
SUPPORTED = {
    "2201": {
        "compressor",
        "sparse_attn_sharedkv",
        "sparse_attn_sharedkv_metadata_host",
    },
    "3510": {
        "compressor",
        "kv_quant_sparse_attn_sharedkv",
        "kv_quant_sparse_attn_sharedkv_metadata_host",
        "kv_compress_epilog",
    },
}


def configure(config, jobs):
    config = dict(config)
    ops = config["ops"]
    if not isinstance(ops, list) or not ops or len(set(ops)) != len(ops):
        raise ValueError("native_ops requires distinct operator names")
    if set(ops) - SUPPORTED[config["arch_version"]] or config.get("has_aicpu", False):
        raise ValueError("Unsupported native operators for selected architecture")
    soc = detect_soc_version()
    targets = {
        ("2201", "ascend910_93"): "Ascend910_9382",
        ("3510", "ascend950"): "Ascend950PR_9599",
    }
    try:
        config["soc"] = targets[(config["arch_version"], soc)]
    except KeyError as error:
        raise RuntimeError(
            "native operators require a matching A3 or A5 target"
        ) from error
    config["jobs"] = jobs or 8
    config["python_package"] = config["name"]
    config["python_bindings"] = True
    config["dispatch_keys"] = {op: DISPATCH_KEYS[op] for op in ops}
    paths = {"include", "csrc/utils", "csrc/batch_matmul_transpose/op_host/tiling"}
    for op in ops:
        paths.add("csrc/" + (op[:-5] if op.endswith("_metadata_host") else op))
    if "kv_quant_sparse_attn_sharedkv" in ops:
        paths.add("csrc/attentions/csrc/ops/utils/inc/kernel")
    config["source_paths"] = sorted(paths)
    config["cmake_sha256"] = hashlib.sha256(
        Path(__file__).with_name("CMakeLists.txt").read_bytes()
    ).hexdigest()
    return config


def prepare_source(
    source, directory, operators=("compressor",), namespace="sgl_native_compressor"
):
    """Keep the kernel/tiling unchanged; fix absent optional shapes in the shim."""
    helper = source / "csrc/utils/ge_helper.h"
    text = helper.read_text()
    old = (
        "const gert::StorageShape *GetOptionalInputShape(uint32_t index) const\n"
        "    {\n        return &inputShape_[index];\n    }"
    )
    fixed = old.replace(
        "return &inputShape_[index];",
        "return inputTensor_[index] == nullptr ? nullptr : &inputShape_[index];",
    )
    if text.count(old) != 1:
        raise RuntimeError("Unexpected pinned GetOptionalInputShape implementation")
    helper.write_text(text.replace(old, fixed, 1))
    registrations = (source / "csrc/pytorch_extensions.cpp").read_text()
    schemas = []
    groups = {}
    for op in operators:
        matches = re.findall(
            r'm\.def\(\s*"' + re.escape(op) + r"\([\s\S]*?\);", registrations
        )
        if len(matches) != 1:
            raise RuntimeError("Expected one pinned native schema for " + op)
        schemas.append(matches[0])
        groups.setdefault(DISPATCH_KEYS[op], []).append(op)
    registration = directory / "registration.cpp"
    registration.write_text(
        "#include <ATen/ATen.h>\n#include <torch/library.h>\n"
        '#include "sgl_kenel_npu_ops.h"\n'
        "TORCH_LIBRARY_FRAGMENT("
        + namespace
        + ", m) {\n"
        + "\n".join(schemas)
        + "\n}\n"
        + "\n".join(
            "TORCH_LIBRARY_IMPL("
            + namespace
            + ", "
            + dispatch
            + ", m) {\n"
            + "\n".join(
                'm.impl("' + op + '", TORCH_FN(sglang::npu_kernel::' + op + "));"
                for op in names
            )
            + "\n}\n"
            for dispatch, names in groups.items()
        )
    )
    return registration


def inspect_vendor(vendor, item):
    """The artifact is a self-contained native registration library, not an OPP."""
    library = Path(vendor) / ("lib" + item["python_package"] + ".so")
    if library.is_symlink() or not library.is_file():
        raise RuntimeError("Missing bundled native library: " + str(library))
    with library.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise RuntimeError("Native registration library is not an ELF library")


def build(item, source, directory):
    registration = prepare_source(source, directory, item["ops"], item["name"])
    native = directory / "native"
    native.mkdir()
    cmake = directory / "cmake"
    run(
        [
            "cmake",
            "-S",
            Path(__file__).parent,
            "-B",
            cmake,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DSGL_ARCH_VERSION=" + item["arch_version"],
            "-DSGL_NATIVE_SOC=" + item["soc"],
            "-DSGL_NATIVE_PACKAGE=" + item["name"],
            "-DSGL_NATIVE_OPERATORS=" + ";".join(item["ops"]),
            "-DSGL_NATIVE_SOURCE=" + str(source),
            "-DSGL_NATIVE_OUTPUT=" + str(native),
            "-DSGL_REGISTRATION_SOURCE=" + str(registration),
        ],
        directory / "configure.log",
    )
    run(
        ["cmake", "--build", cmake, "--parallel", str(item["jobs"])],
        directory / "build.log",
    )
    inspect_vendor(native, item)
    python = directory / "python"
    package = python / item["python_package"]
    (package / "lib").mkdir(parents=True)
    shutil.copy2(native / ("lib" + item["python_package"] + ".so"), package / "lib")
    (package / "__init__.py").write_text(
        "from pathlib import Path\nimport torch\nimport torch_npu\n"
        + 'torch.ops.load_library(str(Path(__file__).parent / "lib/lib'
        + item["name"]
        + '.so"))\n'
        + "\n".join(
            op + " = torch.ops." + item["name"] + "." + op for op in item["ops"]
        )
        + "\n"
    )
    (python / "setup.py").write_text(
        "from setuptools import setup\nfrom setuptools.dist import Distribution\n"
        "class BinaryDistribution(Distribution):\n"
        "    def has_ext_modules(self): return True\n"
        "setup(name=" + repr(item["name"]) + ", version='1.0.0',\n"
        "      packages=[" + repr(item["name"]) + "],\n"
        "      package_data={" + repr(item["name"]) + ": ['lib/*.so']},\n"
        "      distclass=BinaryDistribution)\n"
    )
    run(
        [sys.executable, "setup.py", "bdist_wheel"], directory / "wheel.log", cwd=python
    )
    wheels = list((python / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one precompiled native operator wheel")
    return dict(config=item, vendor=native, wheel=wheels[0], license=source / "LICENSE")


def stage(result, package):
    item = result["config"]
    inspect_vendor(result["vendor"], item)
    destination = package / "_third_party" / item["python_package"]
    extract_python(result["wheel"], item["python_package"], destination)
    inspect_vendor(destination / "lib", item)
    licenses = package / "_third_party/licenses" / item["name"]
    licenses.mkdir(parents=True)
    shutil.copy2(result["license"], licenses / "LICENSE")
    return dict(item, binding="precompiled_native")
