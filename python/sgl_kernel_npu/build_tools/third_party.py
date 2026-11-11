"""Build pinned external operators in isolation and stage their runtime payload."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

from build_tools.cann import PROJECT_ROOT, run

# Explicit allowlist: configuration cannot import arbitrary Python code.
_ADAPTERS = {"ops_transformer", "cann_recipes_infer"}


@lru_cache(maxsize=None)
def get_adapter(name):
    if name not in _ADAPTERS:
        raise ValueError("Unsupported third-party builder: " + name)
    path = PROJECT_ROOT / "third_party" / name / "build.py"
    spec = importlib.util.spec_from_file_location("_sgl_builder_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text())


def detect_cann_version():
    sdk = os.environ.get("ASCEND_HOME_PATH")
    if not sdk:
        raise RuntimeError("Initialize CANN: ASCEND_HOME_PATH is missing")
    for path in (Path(sdk) / "version.cfg", Path(sdk) / "compiler/version.info"):
        if path.is_file():
            match = re.search(
                r"(?im)^\s*(?:version|version_dir)\s*=\s*(\d+\.\d+\.\d+)\s*$",
                path.read_text(),
            )
            if match:
                return match[1]
    raise RuntimeError("Cannot read an exact CANN release from SDK version metadata")


@lru_cache(maxsize=1)
def detect_soc_version():
    """Detect the local build target without importing Torch or initializing NPU."""
    for arguments in (
        ["npu-smi", "info", "-t", "board", "-i", "0"],
        ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"],
    ):
        try:
            output = subprocess.check_output(
                arguments, text=True, stderr=subprocess.STDOUT, timeout=10
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        match = re.search(r"(?im)^\s*Chip Name\s*:\s*(\S+)", output)
        if not match:
            continue
        chip = match[1].lower()
        if chip.startswith(("ascend910b", "910b")):
            return "ascend910b"
        if chip.startswith(("ascend910_93", "910_93")) or chip == "ascend910":
            return "ascend910_93"
        if chip.startswith(("ascend950", "950")):
            return "ascend950"
    raise RuntimeError("Cannot detect build SoC from npu-smi board information")


def resolve_plan(
    version, arch, jobs=None, source_root=PROJECT_ROOT, *, third_party=None
):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected an exact CANN version, e.g. 9.2.0")
    arch_name = {"2201": "arch22", "3510": "arch35"}[str(arch)]
    root = Path(source_root) / "third_party"
    profile = root / "profiles" / ("cann-" + version)
    configs = sorted(profile.glob("*.json"))
    if third_party is not None:
        available = {path.stem for path in configs}
        missing = set(third_party) - available
        if missing:
            raise ValueError(
                "Third-party profiles not found for CANN {}, {}: {} (searched {})".format(
                    version, arch_name, ", ".join(sorted(missing)), profile
                )
            )
        configs = [path for path in configs if path.stem in third_party]
    if not configs:
        print(
            "[INFO] No third-party profiles for CANN {}, {}; "
            "skipping third-party builds (searched {}).".format(
                version, arch_name, profile
            ),
            file=sys.stderr,
            flush=True,
        )
        return []
    builds = []
    for config_path in configs:
        if root.resolve() not in config_path.resolve().parents:
            raise ValueError("Dependency configuration escapes third_party")
        config = read_json(config_path)
        # Profile documentation is not part of the build configuration or cache key.
        config.pop("_comment", None)
        # One upstream profile can require separate regular/experimental vendors.
        additional = config.pop("additional_builds", [])
        builds.append((config_path, config))
        builds.extend((config_path, dict(config, **build)) for build in additional)
    plan = []
    for config_path, config in builds:
        if not isinstance(config.get("has_aicpu", False), bool):
            raise ValueError("has_aicpu must be a boolean")
        if not re.fullmatch(r"[0-9a-f]{40}", config["commit"]):
            raise ValueError("Dependency commit must be a full SHA")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", config["name"]):
            raise ValueError("Invalid dependency identifier: name")
        ops_by_arch = config["ops"]
        if not isinstance(ops_by_arch, dict) or set(ops_by_arch) - {"arch22", "arch35"}:
            raise ValueError("Profile ops must be keyed by arch22 or arch35")
        if any(not isinstance(ops, (list, dict)) for ops in ops_by_arch.values()):
            raise ValueError("Each architecture's ops must be a list or mapping")
        config["ops"] = ops_by_arch.get(arch_name, [])
        if not config["ops"]:
            print(
                "[INFO] No operators for third-party {}, CANN {}, {}; "
                "skipping this library.".format(config_path.stem, version, arch_name),
                file=sys.stderr,
                flush=True,
            )
            continue
        builder = config.get("builder", config_path.stem)
        config["arch_version"] = str(arch)
        config = get_adapter(builder).configure(config, jobs)
        plan.append(
            dict(config, builder=builder, cann_version=version, arch_version=str(arch))
        )
    names = [item["name"] for item in plan]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate dependency names")
    return plan


def extract_python(wheel, original, destination):
    """Extract only the selected import package, never wheel installation hooks."""
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe wheel member: " + item.filename)
            if not path.parts or path.parts[0] != original or item.is_dir():
                continue
            target = destination.joinpath(*path.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
            count += 1
    if not count:
        raise RuntimeError("Python package missing from upstream wheel: " + original)


def build_fingerprint(item):
    """Include the resolved recipe, builder code and relevant build environment."""
    sdk = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    versions = {}
    for name in ("torch", "torch_npu", "setuptools", "wheel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commands = {}
    for command in ("cmake", os.environ.get("CXX", "c++")):
        try:
            commands[command] = subprocess.check_output(
                [command, "--version"], text=True, stderr=subprocess.STDOUT
            )
        except (OSError, subprocess.CalledProcessError):
            commands[command] = None
    scripts = [
        Path(__file__),
        Path(get_adapter(item["builder"]).__file__),
        Path(__file__).with_name("cann.py"),
    ]
    details = {
        "schema": 1,
        "config": item,
        "python": sys.version,
        "machine": platform.machine(),
        "system": platform.platform(),
        "packages": versions,
        "tools": commands,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_HOME_PATH",
                "PATH",
                "LD_LIBRARY_PATH",
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "CMAKE_PREFIX_PATH",
            )
        },
        "sdk": {
            str(path): path.read_text()
            for path in (sdk / "version.cfg", sdk / "compiler/version.info")
            if path.is_file()
        },
        "scripts": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts
        },
    }
    digest = hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
    return digest, details


def _artifact_hashes(result, root):
    """Check all delivered files, including missing/added files and symlink targets."""
    hashes = {}
    for key in ("vendor", "wheel", "license"):
        if key == "wheel" and not result["config"].get("python_bindings", True):
            continue
        path = result[key]
        if not path.exists():
            raise ValueError("Missing cached artifact: " + str(path))
        paths = list(path.rglob("*")) if path.is_dir() else [path]
        for entry in paths:
            if root.resolve() not in entry.resolve().parents:
                raise ValueError("Cached artifact escapes build directory")
            if entry.is_file():
                with entry.open("rb") as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[entry.relative_to(root).as_posix()] = digest.hexdigest()
            elif entry.is_symlink():
                raise ValueError("Invalid cached symlink")
    if not hashes:
        raise ValueError("Empty cached artifacts")
    return hashes


def _load_cached(path, root, item, fingerprint):
    try:
        record = read_json(path)
        if record["fingerprint"] != fingerprint:
            return None
        result = dict(config=item)
        for key in ("vendor", "license") + (
            ("wheel",) if item.get("python_bindings", True) else ()
        ):
            relative = PurePosixPath(record["artifacts"][key])
            if relative.is_absolute() or ".." in relative.parts:
                return None
            result[key] = root / relative
        if _artifact_hashes(result, root) != record["hashes"]:
            return None
        get_adapter(item["builder"]).inspect_vendor(result["vendor"], item)
        return result
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


def build_dependencies(plan, source_root=PROJECT_ROOT):
    """Reuse verified builds; publish a cache record only after full success."""
    if plan and platform.system() != "Linux":
        raise RuntimeError("Third-party CANN compilation requires Linux")
    if not plan:
        return []
    import fcntl

    results = []
    root = Path(source_root) / "build/third_party"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    for item in plan:
        fingerprint, details = build_fingerprint(item)
        record_path = cache / (fingerprint + ".json")
        # Serialize builders of the same recipe across setup processes.
        with (cache / (fingerprint + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = _load_cached(record_path, root, item, fingerprint)
            if result is not None:
                print("Third-party cache hit: " + item["name"], flush=True)
            else:
                print("Third-party cache miss: " + item["name"], flush=True)
                result = _build_dependency(item, root)
                record = {
                    "fingerprint": fingerprint,
                    "inputs": details,
                    "artifacts": {
                        key: result[key].relative_to(root).as_posix()
                        for key in ("vendor", "wheel", "license")
                        if key != "wheel" or item.get("python_bindings", True)
                    },
                    "hashes": _artifact_hashes(result, root),
                }
                temporary = record_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, indent=2) + "\n")
                temporary.replace(record_path)
            results.append(result)
    return results


def _build_dependency(item, root):
    directory = Path(tempfile.mkdtemp(prefix=item["name"] + "-", dir=root))
    source = directory / "source"
    run(
        ["git", "clone", "--no-checkout", item["repository"], source],
        directory / "clone.log",
    )
    run(
        ["git", "checkout", "--detach", item["commit"]],
        directory / "checkout.log",
        cwd=source,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if head != item["commit"]:
        raise RuntimeError("Checked out unexpected dependency commit")
    run(
        ["git", "submodule", "update", "--init", "--recursive"],
        directory / "submodules.log",
        cwd=source,
    )
    return get_adapter(item["builder"]).build(item, source, directory)


def stage_dependencies(results, package):
    manifests = []
    for result in results:
        adapter = get_adapter(result["config"]["builder"])
        manifests.append(adapter.stage(result, package))
    if manifests:
        (package / "_third_party/__init__.py").write_text(
            '"""Bundled third-party Python bindings."""\n'
        )
    return manifests
