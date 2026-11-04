"""Build the repository-owned DSv4.1 DSL sources and their AICPU metadata.

No upstream checkout or vendor Python package is used. Compilation runs in a
separate process so collector imports cannot leak into the wheel coordinator.
"""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from build_tools.cann import PROJECT_ROOT, run

RELATIVE = Path("implementation/cannbot-dsl")
OPERATORS = ("indexer_prologue_k", "mqsmla", "qli", "qsli")
METADATA = (
    (
        "mixed_quant_sparse_flash_mla_metadata",
        "_mqsmla_metadata_kernel",
        "mqsmla_metadata_kernel.so",
    ),
    ("quant_lightning_indexer_metadata", "metadata_kernel", "metadata_kernel.so"),
    (
        "quant_sparse_lightning_indexer_metadata",
        "metadata_kernel",
        "qsli_metadata_kernel.so",
    ),
)


def resolve_plan(version, arch):
    if version != "9.2.0" or str(arch) != "3510":
        raise ValueError("CANNBot-DSL build targets CANN 9.2.0 and --arch-version 3510")
    return {
        "backend": "cannbotdsl",
        "version": "0.8.0",
        "cann_version": version,
        "target": "dav-3510",
        "operators": list(OPERATORS),
    }


def _hashes(directory):
    return {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def build_backend(*, source_root=PROJECT_ROOT):
    from build_tools.third_party import detect_cann_version

    plan = resolve_plan(detect_cann_version(), "3510")
    if sys.platform != "linux" or not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("CANNBot-DSL 0.8.0 requires Linux and Python 3.10-3.12")
    try:
        version = importlib.metadata.version("cannbot-dsl")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install cannbot-dsl==0.8.0 in the build environment"
        ) from error
    if version != plan["version"]:
        raise RuntimeError(f"Expected cannbot-dsl==0.8.0; found {version}")
    dependencies = {}
    for name in ("torch", "torch_npu"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"Install {name} in the build environment before compiling CANNBot-DSL"
            ) from error
    source_root = Path(source_root).resolve()
    source = source_root / "python/sgl_kernel_npu/sgl_kernel_npu" / RELATIVE
    directory = source_root / "build/cannbotdsl"
    directory.mkdir(parents=True, exist_ok=True)
    # A fresh output avoids accidentally packaging binaries from another build.
    work = Path(tempfile.mkdtemp(prefix="ds41-", dir=directory))
    staged = work / "source"
    shutil.copytree(
        source / "attention",
        staged,
        ignore=shutil.ignore_patterns("_native", "_aicpu", "__pycache__"),
    )
    profiles = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    profile_data = Path(profiles).read_bytes() if profiles else None
    env = dict(os.environ)
    if profile_data is not None:
        json.loads(profile_data)
        profile_path = work / "profiles.json"
        profile_path.write_bytes(profile_data)
        env["CANNBOTDSL_DS41_PROFILES"] = str(profile_path)
    # Require-mode is a runtime validation option, not a compiler input.
    env.pop("CANNBOTDSL_NATIVE_BINARY_MODE", None)
    env["PYTHONPATH"] = (
        str(source_root / "python/sgl_kernel_npu")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    output = work / "output"
    run(
        [sys.executable, "-m", "build_tools.cannbotdsl", "--compile", staged, output],
        work / "compile.log",
        env=env,
    )
    manifest = dict(
        plan,
        upstream=json.loads((source / "UPSTREAM.json").read_text()),
        sources=_hashes(staged),
        resources=_hashes(output),
        profiles_sha256=(
            hashlib.sha256(profile_data).hexdigest() if profile_data else None
        ),
        python=sys.version,
        cpu_arch=platform.machine(),
        **dependencies,
    )
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_bundle(output)
    return output


def validate_bundle(output):
    output = Path(output)
    manifest = json.loads((output / "build-manifest.json").read_text())
    plan = resolve_plan("9.2.0", "3510")
    for field, expected_value in plan.items():
        if manifest.get(field) != expected_value:
            raise RuntimeError(
                f"CANNBot-DSL bundle contract mismatch for {field}: "
                f"expected {expected_value!r}, found {manifest.get(field)!r}"
            )
    expected = ["_native/manifest.json", "_native/index.json"]
    expected.extend(f"_native/operators/{name}/operator.json" for name in OPERATORS)
    expected.extend(f"_aicpu/{filename}" for _, _, filename in METADATA)
    for name in expected:
        if name not in manifest["resources"]:
            raise RuntimeError("Missing CANNBot-DSL resource: " + name)
    resources = {k: v for k, v in _hashes(output).items() if k != "build-manifest.json"}
    if resources != manifest["resources"]:
        raise RuntimeError("CANNBot-DSL resource checksum mismatch")
    return manifest


def stage_backend(output, package):
    manifest = validate_bundle(output)
    destination = Path(package) / RELATIVE / "attention"
    # Verify that Native resources are packaged with the exact compiled sources.
    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.glob("*.py")
    }
    expected = {k: v for k, v in manifest["sources"].items() if k.endswith(".py")}
    if actual != expected:
        raise RuntimeError("CANNBot-DSL sources changed after Native compilation")
    for name in ("_native", "_aicpu"):
        shutil.copytree(Path(output) / name, destination / name)
    shutil.copy2(
        Path(output) / "build-manifest.json", destination / "build-manifest.json"
    )
    return manifest


def compile_sources(source, output):
    """Use the 0.8.0 public compiler APIs, as in upstream Native packaging."""
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel
    from cannbotdsl.aot import build

    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True)
    build(
        collectors=source,
        operators=list(OPERATORS),
        target="dav-3510",
        output=output / "_native",
    )
    spec = importlib.util.spec_from_file_location(
        "_sgl_ds41_metadata",
        source / "__init__.py",
        submodule_search_locations=[str(source)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    (output / "_aicpu").mkdir()
    for name, kernel, filename in METADATA:
        module = importlib.import_module(f"_sgl_ds41_metadata.{name}")
        compiled = compile_aicpu_kernel(
            getattr(module, kernel),
            workdir=str(output.parent / "aicpu-work" / name),
            launch_mode="interface",
            npu_arch="dav-3510",
        )
        shutil.copy2(compiled.so_path, output / "_aicpu" / filename)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--compile":
        raise SystemExit("Internal worker: --compile SOURCE OUTPUT")
    compile_sources(sys.argv[2], sys.argv[3])
