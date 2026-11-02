"""Build pinned external operators in isolation and stage their runtime payload."""

import importlib.util
import json
import os
import platform
import re
import subprocess
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

from build_tools.cann import PROJECT_ROOT, run

# Explicit allowlist: configuration cannot import arbitrary Python code.
_ADAPTERS = {"ops_transformer"}


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


def resolve_plan(version, arch, jobs=None, source_root=PROJECT_ROOT):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected an exact CANN version, e.g. 9.2.0")
    arch_name = {"2201": "arch22", "3510": "arch35"}[str(arch)]
    root = Path(source_root) / "third_party"
    profile = root / "profiles" / ("cann-" + version) / arch_name
    configs = sorted(profile.glob("*.json"))
    if not configs:
        raise ValueError(
            "No third-party profile: {} (use --without-third-party for native-only builds)".format(
                profile
            )
        )
    plan = []
    for config_path in configs:
        if root.resolve() not in config_path.resolve().parents:
            raise ValueError("Dependency configuration escapes third_party")
        config = read_json(config_path)
        if not isinstance(config.get("has_aicpu", False), bool):
            raise ValueError("has_aicpu must be a boolean")
        if not re.fullmatch(r"[0-9a-f]{40}", config["commit"]):
            raise ValueError("Dependency commit must be a full SHA")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", config["name"]):
            raise ValueError("Invalid dependency identifier: name")
        builder = config.get("builder", config_path.stem)
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


def build_dependencies(plan, source_root=PROJECT_ROOT):
    """Fresh isolated builds; cache reuse is deliberately deferred in this version."""
    if plan and platform.system() != "Linux":
        raise RuntimeError("Third-party CANN compilation requires Linux")
    results = []
    root = Path(source_root) / "build/third_party"
    root.mkdir(parents=True, exist_ok=True)
    for item in plan:
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
        results.append(get_adapter(item["builder"]).build(item, source, directory))
    return results


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
