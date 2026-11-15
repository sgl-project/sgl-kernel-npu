#!/usr/bin/env python3
"""Prepare a minimal, pinned CANN OPP source tree for the original LIM baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

UPSTREAM_REVISION = "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e"
UPSTREAM_REPOSITORY = "vllm-project/vllm-ascend"
SOURCE_ARCHIVE = "original-lim-pr18383-source.tar.xz"
SOURCE_ARCHIVE_SHA256 = (
    "5bfe5edd5c7b0a9f76001b9aaf93c86ffb7ebe4f9c790d675eb5e8fe0a335fff"
)
DEPENDENCY_ARCHIVES = {
    "json": {
        "path": "vendor/json-include-3.11.3.zip",
        "destination": "source/csrc/third_party/pkg/include.zip",
        "license_path": "vendor/json-LICENSE-MIT.txt",
        "license_destination": "source/csrc/third_party/licenses/nlohmann-json-LICENSE.MIT",
        "version": "3.11.3",
        "source": "https://github.com/nlohmann/json/tree/v3.11.3/include",
        "sha256": "5580533576f6cde8f2c36843f3b27fc552d263bdcc00c597a36b1919a4a78df8",
    },
    "abseil_cpp": {
        "path": "vendor/abseil-cpp-20230802.1.tar.gz",
        "destination": "source/csrc/third_party/pkg/abseil-cpp-20230802.1.tar.gz",
        "version": "20230802.1",
        "source": "https://github.com/abseil/abseil-cpp/tree/20230802.1",
        "sha256": "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed",
    },
    "protobuf": {
        "path": "vendor/protobuf-25.1.tar.gz",
        "destination": "source/csrc/third_party/pkg/protobuf-25.1.tar.gz",
        "version": "25.1",
        "source": "https://github.com/protocolbuffers/protobuf/tree/v25.1",
        "sha256": "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42",
    },
    "makeself": {
        "path": "vendor/makeself-upstream-2.5.0.tar.gz",
        "destination": "source/csrc/third_party/makeself",
        "version": "2.5.0 (upstream release; CANN patch archive unavailable from this host)",
        "source": "https://github.com/megastep/makeself/tree/release-2.5.0",
        "sha256": "705d0376db9109a8ef1d4f3876c9997ee6bed454a23619e1dbc03d25033e46ea",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_archive_path(member_name: str) -> PurePosixPath:
    """Validate an archive member name and return its normalized relative path."""
    path = PurePosixPath(member_name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {member_name!r}")
    if "\\" in member_name or ":" in member_name:
        raise ValueError(f"unsafe archive path: {member_name!r}")
    return path


def _safe_target(root: Path, relative: PurePosixPath) -> Path:
    target = root.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"archive path escapes output root: {relative}")
    return target


def _extract_source(archive_path: Path, source_root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            rel = safe_archive_path(member.name)
            if not rel.parts or not (
                rel.parts[0] == "csrc" or rel.parts == ("LICENSE",)
            ):
                raise ValueError(f"unexpected source archive member: {member.name!r}")
            if member.isdir():
                _safe_target(source_root, rel).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"non-regular source archive member rejected: {member.name!r}"
                )
            target = _safe_target(source_root, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read source archive member: {member.name!r}")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            target.chmod(member.mode & 0o777)
            files.append(
                {
                    "path": f"source/{rel.as_posix()}",
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
    return sorted(files, key=lambda item: item["path"])


def _copy_or_extract_dependencies(
    package_root: Path, staging_root: Path
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, spec in DEPENDENCY_ARCHIVES.items():
        artifact = package_root / spec["path"]
        if not artifact.is_file():
            raise FileNotFoundError(f"vendored dependency is missing: {artifact}")
        digest = sha256_file(artifact)
        if digest != spec["sha256"]:
            raise ValueError(f"vendored dependency SHA256 mismatch for {artifact}")
        rel_destination = PurePosixPath(spec["destination"])
        target = _safe_target(staging_root, rel_destination)
        if name != "makeself":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact, target)
            license_entry = None
            if spec.get("license_path"):
                license_src = package_root / spec["license_path"]
                license_dst = _safe_target(
                    staging_root, PurePosixPath(spec["license_destination"])
                )
                license_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(license_src, license_dst)
                license_entry = {
                    "path": spec["license_destination"],
                    "size_bytes": license_dst.stat().st_size,
                    "sha256": sha256_file(license_dst),
                }
            extracted_headers = []
            if name == "json":
                header_root = _safe_target(
                    staging_root, PurePosixPath("source/csrc/third_party/json")
                )
                with zipfile.ZipFile(artifact) as archive:
                    for member in archive.infolist():
                        member_path = member.filename.rstrip("/")
                        rel = safe_archive_path(member_path)
                        if not rel.parts or rel.parts[0] != "include":
                            raise ValueError(
                                f"unexpected JSON header member: {member.filename!r}"
                            )
                        inner = rel
                        output_path = _safe_target(header_root, inner)
                        if member.is_dir():
                            output_path.mkdir(parents=True, exist_ok=True)
                            continue
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, output_path.open(
                            "xb"
                        ) as output:
                            shutil.copyfileobj(source, output)
                        extracted_headers.append(
                            {
                                "path": f"source/csrc/third_party/json/{inner.as_posix()}",
                                "size_bytes": output_path.stat().st_size,
                                "sha256": sha256_file(output_path),
                            }
                        )
                if not (header_root / "include/nlohmann/json.hpp").is_file():
                    raise ValueError(
                        "JSON include archive is missing nlohmann/json.hpp"
                    )
            result.append(
                {
                    "name": name,
                    "version": spec["version"],
                    "source": spec["source"],
                    "path": rel_destination.as_posix(),
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                    **({"license_file": license_entry} if license_entry else {}),
                    **(
                        {"unpacked_headers": extracted_headers}
                        if extracted_headers
                        else {}
                    ),
                }
            )
            continue

        target.mkdir(parents=True, exist_ok=True)
        extracted: list[dict[str, Any]] = []
        with tarfile.open(artifact, "r:gz") as archive:
            for member in archive.getmembers():
                rel = safe_archive_path(member.name)
                if member.isdir() and rel.parts == ("makeself-release-2.5.0",):
                    continue
                if len(rel.parts) < 2 or rel.parts[0] != "makeself-release-2.5.0":
                    raise ValueError(f"unexpected makeself member: {member.name!r}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"non-regular makeself member rejected: {member.name!r}"
                    )
                inner = PurePosixPath(*rel.parts[1:])
                output_path = _safe_target(target, inner)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read makeself member: {member.name!r}")
                with stream, output_path.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                output_path.chmod(member.mode & 0o777)
                extracted.append(
                    {
                        "path": f"{rel_destination.as_posix()}/{inner.as_posix()}",
                        "size_bytes": output_path.stat().st_size,
                        "sha256": sha256_file(output_path),
                    }
                )
        if (
            not (target / "makeself.sh").is_file()
            or not (target / "makeself-header.sh").is_file()
        ):
            raise ValueError("makeself archive does not contain expected scripts")
        result.append(
            {
                "name": name,
                "version": spec["version"],
                "source": spec["source"],
                "archive_sha256": digest,
                "installed_root": rel_destination.as_posix(),
                "files": sorted(extracted, key=lambda item: item["path"]),
                "note": "CANN's pinned patched makeself tarball was unavailable (GitCode returned HTTP 401); upstream 2.5.0 is preinstalled to avoid a build-time download.",
            }
        )
    return result


def prepare(output_root: Path, operator: str = "lim") -> Path:
    if operator != "lim":
        raise ValueError("only --operator lim is currently packaged")
    output_root = output_root.absolute()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output root: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parent
    source_archive = package_root / "vendor" / SOURCE_ARCHIVE
    if not source_archive.is_file():
        raise FileNotFoundError(f"pinned source archive is missing: {source_archive}")
    if sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError(f"pinned source archive SHA256 mismatch: {source_archive}")

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.prepare-", dir=output_root.parent)
    )
    try:
        source_root = staging_root / "source"
        source_root.mkdir()
        source_files = _extract_source(source_archive, source_root)
        vendored_dependencies = _copy_or_extract_dependencies(
            package_root, staging_root
        )
        # This upstream snapshot unconditionally adds these optional categories
        # from its root CMakeLists.txt although the directories are absent at the
        # pinned revision. Empty CMake projects keep the CANN builder usable for
        # the single selected LIM operator without changing any upstream file.
        cmake_shims = []
        for category in ("ffn", "posembedding"):
            shim = source_root / "csrc" / category / "CMakeLists.txt"
            shim.parent.mkdir(parents=True, exist_ok=True)
            shim.write_text(
                "# Empty category shim for the pinned single-operator build.\n",
                encoding="utf-8",
            )
            cmake_shims.append(
                {
                    "path": f"source/csrc/{category}/CMakeLists.txt",
                    "purpose": "Satisfy unconditional add_subdirectory in the pinned upstream CMake; category is not selected for this build.",
                    "sha256": sha256_file(shim),
                }
            )
        manifest = {
            "schema_version": 1,
            "operator": operator,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "revision": UPSTREAM_REVISION,
            },
            "source_root": "source",
            "csrc_root": "source/csrc",
            "files": source_files,
            "packaging_shims": cmake_shims,
            "vendored_dependencies": vendored_dependencies,
        }
        manifest_path = staging_root / "SOURCE_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging_root, output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return output_root / "SOURCE_MANIFEST.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--operator", choices=("lim",), default="lim")
    args = parser.parse_args()
    manifest = prepare(args.output_root, args.operator)
    print(f"prepared original {args.operator} source: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
