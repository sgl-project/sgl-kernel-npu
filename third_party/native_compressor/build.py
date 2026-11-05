"""Bundle the pinned, fused native compressor; never use an installed extension."""

import hashlib
import re
import shutil
import sys
from pathlib import Path

from build_tools.cann import run
from build_tools.third_party import detect_soc_version, extract_python


def configure(config, jobs):
    config = dict(config)
    if config["ops"] != ["compressor"] or config.get("has_aicpu", False):
        raise ValueError("native_compressor builds only the fused compressor")
    soc = detect_soc_version()
    targets = {
        ("2201", "ascend910_93"): "Ascend910_9382",
        ("3510", "ascend950"): "Ascend950PR_9599",
    }
    try:
        config["soc"] = targets[(config["arch_version"], soc)]
    except KeyError as error:
        raise RuntimeError(
            "native compressor requires a matching A3 or A5 target"
        ) from error
    config["jobs"] = jobs or 8
    config["python_package"] = config["name"]
    config["python_bindings"] = True
    # Include the CMake recipe in the existing dependency cache fingerprint.
    config["cmake_sha256"] = hashlib.sha256(
        Path(__file__).with_name("CMakeLists.txt").read_bytes()
    ).hexdigest()
    return config


def prepare_source(source, directory):
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
    schemas = re.findall(r'm\.def\(\s*"compressor\([\s\S]*?\);', registrations)
    if len(schemas) != 1:
        raise RuntimeError("Expected one pinned native compressor schema")
    registration = directory / "registration.cpp"
    registration.write_text(
        "#include <ATen/ATen.h>\n#include <torch/library.h>\n"
        '#include "sgl_kenel_npu_ops.h"\n'
        "TORCH_LIBRARY_FRAGMENT(sgl_native_compressor, m) {\n"
        + schemas[0]
        + "\n}\nTORCH_LIBRARY_IMPL(sgl_native_compressor, PrivateUse1, m) {\n"
        'm.impl("compressor", TORCH_FN(sglang::npu_kernel::compressor));\n}\n'
    )
    return registration


def inspect_vendor(vendor, item):
    """The artifact is a self-contained native registration library, not an OPP."""
    library = Path(vendor) / "libsgl_native_compressor.so"
    if library.is_symlink() or not library.is_file():
        raise RuntimeError("Missing bundled native compressor: " + str(library))
    with library.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise RuntimeError("Native compressor is not an ELF library")


def build(item, source, directory):
    registration = prepare_source(source, directory)
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
            "-DSGL_COMPRESSOR_SOC=" + item["soc"],
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
    shutil.copy2(native / "libsgl_native_compressor.so", package / "lib")
    (package / "__init__.py").write_text(
        "from pathlib import Path\nimport torch\nimport torch_npu\n"
        'torch.ops.load_library(str(Path(__file__).parent / "lib/libsgl_native_compressor.so"))\n'
        "compressor = torch.ops.sgl_native_compressor.compressor\n"
    )
    (python / "setup.py").write_text(
        "from setuptools import setup\nfrom setuptools.dist import Distribution\n"
        "class BinaryDistribution(Distribution):\n"
        "    def has_ext_modules(self): return True\n"
        "setup(name='sgl_native_compressor', version='1.0.0',\n"
        "      packages=['sgl_native_compressor'],\n"
        "      package_data={'sgl_native_compressor': ['lib/*.so']},\n"
        "      distclass=BinaryDistribution)\n"
    )
    run(
        [sys.executable, "setup.py", "bdist_wheel"], directory / "wheel.log", cwd=python
    )
    wheels = list((python / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one precompiled native compressor wheel")
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
