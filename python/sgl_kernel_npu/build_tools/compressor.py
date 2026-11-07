"""Build the existing fused compressor without changing its C++ sources."""

import sys
from pathlib import Path

from build_tools.cann import _cache, run
from build_tools.third_party import detect_soc_version


def build_compressor(directory, arch_version, *, source_root, jobs=None):
    directory = Path(directory)
    if directory.is_symlink():
        raise RuntimeError("Refusing a symlinked compressor build directory")
    expected = {
        "CMAKE_HOME_DIRECTORY": str(
            source_root / "python/sgl_kernel_npu/build_tools/compressor"
        ),
        "SGL_SOURCE_ROOT": str(source_root),
        "SGL_ARCH_VERSION": arch_version,
        "SOC_VERSION": {
            "ascend910b": "Ascend910B1",
            "ascend910_93": "Ascend910_9382",
            "ascend950": "Ascend950PR_9599",
        }[detect_soc_version()],
        "Python3_EXECUTABLE": sys.executable,
    }
    cached = _cache(directory)
    if cached and any(cached.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Compressor configuration changed; rerun with --make-clean")
    # Reuse the parent Torch build's validated environment signature.
    environment = directory.parent / "torch/torch-environment.json"
    signature = directory / "environment.json"
    current = environment.read_text()
    if signature.exists() and signature.read_text() != current:
        raise RuntimeError("Compressor environment changed; rerun with --make-clean")
    directory.mkdir(parents=True, exist_ok=True)
    signature.write_text(current)
    run(
        [
            "cmake",
            "-S",
            expected["CMAKE_HOME_DIRECTORY"],
            "-B",
            directory,
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        + [
            "-D" + key + "=" + value
            for key, value in expected.items()
            if key != "CMAKE_HOME_DIRECTORY"
        ],
        directory / "configure.log",
        cwd=source_root,
    )
    command = ["cmake", "--build", directory]
    if jobs is not None:
        command += ["--parallel", str(jobs)]
    run(command, directory / "build.log", cwd=source_root)
    library = directory / "lib/libsgl_compressor.so"
    if not library.is_file() or not library.stat().st_size:
        raise RuntimeError("Missing fused compressor library: " + str(library))
    return library
