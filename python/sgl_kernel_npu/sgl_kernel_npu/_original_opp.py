"""Activate the wheel's pinned original OPP without initializing an NPU."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

REVISIONS = {
    "lim": "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e",
}


def activate(package_root: Path) -> Optional[dict]:
    """Verify and prepend the bundled vendor before torch_npu is imported."""
    root = Path(package_root).resolve() / "lib/a5_original_opp"
    manifest_path = root / "runtime_manifest.json"
    if not manifest_path.exists():
        return None
    if root.is_symlink():
        raise ImportError("original A5 OPP bundle must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("backend") != "original":
        raise ImportError("invalid original A5 OPP backend manifest")
    for operator, revision in REVISIONS.items():
        if manifest.get("operators", {}).get(operator, {}).get("revision") != revision:
            raise ImportError(f"unexpected original A5 {operator} revision")

    def checked(relative: str) -> Path:
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root.resolve())
        except ValueError:
            raise ImportError("original A5 OPP manifest path escapes bundle")
        return path

    vendor = checked(manifest["vendor"])
    files = manifest.get("files", {})
    if not files or not any(
        "/op_api/lib/" in relative and relative.endswith(".so") for relative in files
    ):
        raise ImportError("original A5 OPP manifest lacks ACLNN libraries")
    for relative, expected in files.items():
        path = checked(relative)
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(data)
        if checksum.hexdigest() != expected:
            raise ImportError(f"original A5 OPP bundle SHA256 mismatch: {relative}")
    existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    selected = str(vendor)
    paths = list(
        dict.fromkeys([selected] + [p for p in existing if p and p != selected])
    )
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(paths)
    return manifest


def verify_library(package_root: Path, manifest: dict, library: str) -> None:
    """Check that a resolved ACLNN DSO is a recorded, intact bundled file."""
    root = (Path(package_root).resolve() / "lib/a5_original_opp").resolve()
    vendor = (root / manifest["vendor"]).resolve()
    selected = Path(library).resolve(strict=True)
    try:
        selected.relative_to(vendor)
        relative = selected.relative_to(root).as_posix()
    except ValueError:
        raise ImportError(
            f"original A5 ACLNN library resolved outside bundled OPP: {selected}"
        )
    expected = manifest["files"].get(relative)
    if expected is None:
        raise ImportError(
            f"resolved original A5 ACLNN library is not recorded: {selected}"
        )
    checksum = hashlib.sha256()
    with selected.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(data)
    if checksum.hexdigest() != expected:
        raise ImportError(
            f"resolved original A5 ACLNN library SHA256 mismatch: {selected}"
        )
