"""Shared build paths, command execution and cache helpers; no backend imports."""

import json
import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def architecture(arch_version, source_root=PROJECT_ROOT):
    architectures = json.loads(
        (Path(source_root) / "csrc/cmake/architectures.json").read_text()
    )
    if str(arch_version) not in architectures:
        raise ValueError("--arch-version must be 2201 or 3510")
    return architectures[str(arch_version)]


def run(command, log_path, *, cwd=None, env=None):
    command = [str(value) for value in command]
    display = " ".join(shlex.quote(value) for value in command)
    print("\n$ " + display, flush=True)
    with log_path.open("w") as log:
        log.write(display + "\n")
        log.flush()
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            result = process.wait()
    if result:
        raise RuntimeError("Command failed ({}); see {}".format(result, log_path))


def _remove_tree(path):
    if path.is_symlink():
        raise RuntimeError(
            "Refusing to clean a symlinked build directory: {}".format(path)
        )
    if not path.exists():
        return

    # SDK installers may make vendor directories read-only. Do not follow links.
    for directory, _, _ in os.walk(path, followlinks=False):
        entry = Path(directory)
        entry.chmod(entry.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    shutil.rmtree(path)


def _cache(directory):
    values = {}
    path = directory / "CMakeCache.txt"
    if path.is_file():
        for line in path.read_text().splitlines():
            if (
                not line.startswith(("//", "#"))
                and "=" in line
                and ":" in line.split("=", 1)[0]
            ):
                key, value = line.split("=", 1)
                values[key.split(":", 1)[0]] = value
    return values


def _selection(directory, name="selected-operators.txt"):
    path = directory / name
    return tuple(filter(None, path.read_text().splitlines())) if path.is_file() else ()


def _cache_list(values, key):
    if key not in values:
        raise RuntimeError("CMake did not export required build information: " + key)
    return tuple(filter(None, values[key].split(";")))


def parse_operators(value):
    if value == "all":
        return ("ALL",)
    operators = tuple(part.strip() for part in value.split(","))
    if not operators or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", item) for item in operators
    ):
        raise ValueError("--ops must be 'all' or comma-separated operator IDs")
    if (
        "all" in operators
        or "ALL" in operators
        or len(set(operators)) != len(operators)
    ):
        raise ValueError(
            "--ops must not mix 'all' with IDs, use reserved 'ALL', or repeat an ID"
        )
    return operators
