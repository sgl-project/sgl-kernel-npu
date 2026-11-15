#!/usr/bin/env python3
"""Stage a private original OPP installation into the normal kernel wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

VENDOR = "vendors/sglc8original_transformer"
REVISIONS = {
    "lim": "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e",
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(data)
    return checksum.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path stays below root on Python 3.7+."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def source_receipts(metadata: dict):
    """Yield every prepared source/dependency file recorded by the manifest."""
    for entry in metadata.get("files", []):
        yield entry
    for entry in metadata.get("packaging_shims", []):
        yield entry
    for dependency in metadata.get("vendored_dependencies", []):
        if "path" in dependency and "sha256" in dependency:
            yield dependency
        license_entry = dependency.get("license_file")
        if license_entry:
            yield license_entry
        for key in ("unpacked_headers", "files"):
            for entry in dependency.get(key, []):
                yield entry


def stage(build_root: Path, package_root: Path) -> Path:
    build_root = build_root.resolve(strict=True)
    source_manifest = build_root / "prepared/SOURCE_MANIFEST.json"
    metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    for operator, revision in REVISIONS.items():
        if metadata.get("operators", {}).get(operator, {}).get("revision") != revision:
            raise ValueError(f"unexpected original {operator} revision")
    recorded_sources = {}
    for entry in source_receipts(metadata):
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not relative or not expected:
            raise ValueError("original source manifest contains an incomplete receipt")
        if relative in recorded_sources and recorded_sources[relative] != expected:
            raise ValueError(f"conflicting original source receipts: {relative}")
        recorded_sources[relative] = expected
    if not recorded_sources:
        raise ValueError("original source manifest contains no file receipts")
    for relative, expected in recorded_sources.items():
        path = (build_root / "prepared" / relative).resolve(strict=True)
        if not is_within(path, build_root / "prepared"):
            raise ValueError("original source receipt path escapes preparation")
        if digest(path) != expected:
            raise ValueError(f"original source changed after preparation: {relative}")
    installed = build_root / "opp" / VENDOR
    if not installed.is_dir() or installed.is_symlink():
        raise ValueError(f"original OPP vendor installation is missing: {installed}")
    if not list((installed / "op_api/lib").glob("*.so")):
        raise ValueError("original OPP installation has no ACLNN shared library")
    receipt = build_root / "installed_files.sha256.json"
    installed_hashes = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(installed_hashes, dict) or not installed_hashes:
        raise ValueError("original OPP installation receipt is empty")
    link_receipt = build_root / "installed_symlinks.json"
    installed_links = (
        json.loads(link_receipt.read_text(encoding="utf-8"))
        if link_receipt.is_file()
        else {}
    )
    actual_files = {
        path.relative_to(build_root / "opp").as_posix()
        for path in installed.rglob("*")
        if path.is_file()
    }
    actual_links = {
        path.relative_to(build_root / "opp").as_posix()
        for path in installed.rglob("*")
        if path.is_symlink()
    }
    if actual_files != set(installed_hashes):
        raise ValueError("original OPP file receipt does not match the installation")
    if actual_links != set(installed_links):
        raise ValueError("original OPP link receipt does not match the installation")
    for path in installed.rglob("*"):
        if path.is_symlink():
            relative = path.relative_to(build_root / "opp").as_posix()
            if (
                installed_links.get(relative) != os.readlink(path)
                or os.path.isabs(os.readlink(path))
                or not is_within(path.resolve(strict=True), installed)
            ):
                raise ValueError(
                    f"unrecorded or non-relocatable original OPP link: {relative}"
                )
    for relative, expected in installed_hashes.items():
        path = (build_root / "opp" / relative).resolve(strict=True)
        if not is_within(path, installed):
            raise ValueError("original OPP receipt path escapes installation")
        if digest(path) != expected:
            raise ValueError(f"original OPP installation changed: {relative}")
    package = package_root.resolve()
    lib = package / "lib"
    if lib.is_symlink() or lib.resolve().parent != package:
        raise ValueError("package lib must stay inside the kernel package")
    destination = lib / "a5_original_opp"
    if destination.is_symlink() or not is_within(destination, lib):
        raise ValueError("refusing to replace an OPP path outside the package lib")
    if destination.exists():
        shutil.rmtree(destination)
    target = destination / VENDOR
    # Wheel archives do not preserve filesystem symlinks reliably. Materialize
    # each validated internal target without changing its bytes or visible path.
    shutil.copytree(installed, target, symlinks=False)
    shutil.copyfile(source_manifest, destination / "SOURCE_MANIFEST.json")
    files = {
        path.relative_to(destination).as_posix(): digest(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "backend": "original",
        "vendor": VENDOR,
        "operators": metadata["operators"],
        "operator_backends": {"lim": "original_aclnn", "copy_sfa": "direct"},
        "files": files,
        "source_symlinks": installed_links,
        "source_manifest_sha256": digest(source_manifest),
        "execution_configuration": {
            "SIMT_POSTPROCESS": 1,
            "SIMT_VICTIM_UNPACK": 1,
            "SIMT_BATCHED_REPAIR": 1,
            "user_ub_bytes": 216 * 1024,
            "scope": "pinned source defaults; actual target execution requires A5 validation",
        },
    }
    result = destination / "runtime_manifest.json"
    result.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return result


def clear(package_root: Path) -> None:
    package = package_root.resolve()
    lib = package / "lib"
    if lib.is_symlink() or lib.resolve().parent != package:
        raise ValueError("package lib must stay inside the kernel package")
    destination = lib / "a5_original_opp"
    if destination.is_symlink() or not is_within(destination, lib):
        raise ValueError("refusing to clear an OPP path outside the package lib")
    if destination.exists():
        shutil.rmtree(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-root", type=Path)
    mode.add_argument("--clear", action="store_true")
    parser.add_argument("--package-root", required=True, type=Path)
    args = parser.parse_args()
    if args.clear:
        clear(args.package_root)
    else:
        print(stage(args.build_root, args.package_root))
