#!/usr/bin/env python3
"""Record a completed private OPP installation, including its internal links."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path stays below root on Python 3.7+."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def record(build_root: Path) -> dict:
    root = build_root.resolve(strict=True)
    opp = root / "opp"
    vendor = (opp / "vendors/sglc8original_transformer").resolve(strict=True)
    files = {}
    links = {}
    for path in sorted(opp.rglob("*")):
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not is_within(
                path.resolve(strict=True), vendor
            ):
                raise ValueError(
                    f"OPP link must stay inside the relocatable vendor: {path}"
                )
            links[path.relative_to(opp).as_posix()] = target
        if path.is_file():
            if not is_within(path.resolve(strict=True), vendor):
                raise ValueError(
                    f"OPP file must stay inside the private vendor: {path}"
                )
            files[path.relative_to(opp).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if not files:
        raise ValueError("private OPP installation is empty")
    (root / "installed_files.sha256.json").write_text(
        json.dumps(files, indent=2) + "\n"
    )
    (root / "installed_symlinks.json").write_text(json.dumps(links, indent=2) + "\n")
    return {"files": len(files), "internal_symlinks": len(links)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", required=True, type=Path)
    print(json.dumps(record(parser.parse_args().build_root), indent=2))
