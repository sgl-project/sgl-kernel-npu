"""Check an arm interpreter without importing or initializing NPU libraries."""

from __future__ import annotations

import subprocess


def check_interpreter(executable: str, environment=None) -> None:
    probe = (
        "import importlib.util,sys; "
        "missing=[name for name in ('torch','torch_npu') "
        "if importlib.util.find_spec(name) is None]; "
        "print('Python: '+sys.executable); "
        "print('Missing modules: '+', '.join(missing)) if missing else None; "
        "sys.exit(bool(missing))"
    )
    try:
        result = subprocess.run(
            [executable, "-c", probe],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot check arm Python {executable}: {error}") from error
    if result.returncode:
        raise RuntimeError(
            f"Arm Python {executable} cannot locate the required runtime. "
            "Use the container's Torch-enabled Python to prepare arms.json.\n"
            + result.stdout
            + result.stderr
        )
