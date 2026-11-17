"""Unpack pre/post wheels into private prefixes without changing installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

from runtime_check import check_interpreter


def unpack(wheel: Path, destination: Path) -> dict:
    wheel = wheel.resolve(strict=True)
    if destination.exists():
        raise ValueError(f"refusing to overwrite prefix: {destination}")
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            raw_name = entry.orig_filename
            path = PurePosixPath(raw_name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in raw_name
                or "\x00" in raw_name
                or (path.parts and ":" in path.parts[0])
            ):
                raise ValueError(f"unsafe wheel entry: {entry.filename}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"wheel symlinks are unsupported: {entry.filename}")
        if "sgl_kernel_npu/__init__.py" not in archive.namelist():
            raise ValueError("not an sgl-kernel-npu wheel")
        destination.mkdir(parents=True)
        archive.extractall(destination)
        for entry in archive.infolist():
            mode = (entry.external_attr >> 16) & 0o777
            if mode and not entry.is_dir():
                (destination / entry.filename).chmod(mode)
    return dict(
        wheel=str(wheel),
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        prefix=str(destination.resolve()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-wheel", type=Path, required=True)
    parser.add_argument("--post-wheel", type=Path, required=True)
    parser.add_argument("--framework-prefix", type=Path, required=True)
    parser.add_argument("--pre-framework-prefix", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        parser.error(f"refusing to overwrite output root: {root}")
    framework = args.framework_prefix.resolve(strict=True)
    pre_framework = (args.pre_framework_prefix or framework).resolve(strict=True)
    check_interpreter(sys.executable)
    args.pre_wheel.resolve(strict=True)
    args.post_wheel.resolve(strict=True)
    pre = unpack(args.pre_wheel, root / "pre")
    post = unpack(args.post_wheel, root / "post")
    arms = []
    for name, kind, source, prefix in (
        ("native", "native", framework, post["prefix"]),
        ("pre_migration", "lim_copysfa", pre_framework, pre["prefix"]),
        ("post_migration", "lim_copysfa", framework, post["prefix"]),
    ):
        arms.append(
            dict(
                name=name,
                kind=kind,
                python=sys.executable,
                framework_prefix=str(source),
                kernel_prefix=prefix,
                libraries=[],
            )
        )
    (root / "arms.json").write_text(
        json.dumps(dict(schema_version=1, arms=arms), indent=2) + "\n"
    )
    (root / "wheel-manifest.json").write_text(
        json.dumps(dict(pre=pre, post=post), indent=2) + "\n"
    )
    print(root / "arms.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
