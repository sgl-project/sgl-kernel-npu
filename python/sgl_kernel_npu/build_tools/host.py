"""Compile selected HOST implementations in the configured shared Torch project."""

from pathlib import Path

from build_tools.common import _cache, run


def build_host(arch_version, operators, *, directory, jobs=None):
    """Build only HOST implementation objects; no CANN input, adapters or linking."""
    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be positive")
    operators = tuple(operators)
    if not operators:
        raise ValueError("No HOST operators selected; skip build_host")
    directory = Path(directory).absolute()
    if directory.is_symlink():
        raise RuntimeError("Refusing a symlinked Torch build directory")
    cached = _cache(directory)
    if cached.get("SGL_ARCH_VERSION") != str(arch_version) or cached.get(
        "SGL_HOST_OPERATORS"
    ) != ";".join(operators):
        raise RuntimeError("Configured HOST build changed; rerun with --make-clean")
    command = ["cmake", "--build", directory, "--target", "sgl_host_operators"]
    if jobs is not None:
        command += ["--parallel", str(jobs)]
    run(command, directory / "host-build.log")
    print("HOST_OBJECTS: " + ",".join(operators), flush=True)
