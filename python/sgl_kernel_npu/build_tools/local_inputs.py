"""Hash repository-owned files used to build external dependencies."""

import hashlib
from pathlib import Path, PurePosixPath

from build_tools.cann import PROJECT_ROOT


def local_input_hashes(paths, root=None):
    """Hash declared local inputs without following links or leaving the repo.

    Paths are repository-relative. Adapters can add the returned mapping to
    their resolved configuration so the dependency build cache is invalidated
    when an overlay file is added, removed, or changed.
    """
    root = Path(PROJECT_ROOT if root is None else root)
    hashes = {}
    for value in paths:
        if not isinstance(value, str) or "\\" in value:
            raise ValueError("Invalid local build input: " + str(value))
        relative = PurePosixPath(value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Invalid local build input: " + str(value))
        current = root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("Symlinked local build input: " + str(current))
        if not current.exists():
            raise ValueError("Missing local build input: " + str(current))
        entries = sorted(current.rglob("*")) if current.is_dir() else [current]
        for entry in entries:
            if entry.is_symlink():
                raise ValueError("Symlinked local build input: " + str(entry))
            if entry.is_file():
                key = entry.relative_to(root).as_posix()
                hashes[key] = hashlib.sha256(entry.read_bytes()).hexdigest()
            elif not entry.is_dir():
                raise ValueError("Invalid local build input: " + str(entry))
    return hashes
