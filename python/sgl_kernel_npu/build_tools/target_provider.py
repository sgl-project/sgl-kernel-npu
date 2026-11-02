"""Build-time staging of target-specific operator providers.

Providers live under ``target_providers/<target>/`` and are copied into the
wheel's ``sgl_kernel_npu`` package at build time. The relative module path is
the registration key: ``target_providers/Ascend910/norm/gemma_rmsnorm.py``
becomes ``sgl_kernel_npu/norm/gemma_rmsnorm.py``. The build system knows
targets, but does not know operators.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SUPPORTED_PROVIDER_TARGETS = {
    "Ascend910",
    "Ascend950",
}


def stage_target_providers(
    source_root: Path,
    build_lib: Path,
    target: str,
) -> None:
    """Stage every provider for ``target`` into ``build_lib/sgl_kernel_npu``.

    ``source_root`` is the directory that contains ``target_providers/``.
    """
    if target not in SUPPORTED_PROVIDER_TARGETS:
        raise ValueError(f"Unsupported provider target: {target}")

    provider_root = source_root / "target_providers" / target

    if not provider_root.is_dir():
        raise RuntimeError(
            f"Missing provider directory for target {target}: {provider_root}"
        )

    package_root = build_lib / "sgl_kernel_npu"

    for src in sorted(provider_root.rglob("*.py")):
        relative_path = src.relative_to(provider_root)
        dst = package_root / relative_path

        _validate_destination(dst, relative_path)

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _validate_destination(dst: Path, relative_path: Path) -> None:
    if dst.exists():
        raise RuntimeError(
            "Target-specific provider conflicts with an existing common "
            f"module: {relative_path}"
        )


def collect_provider_modules(root: Path, target: str) -> set[Path]:
    """Return the relative module paths offered by ``target``.

    Used by CI to assert that every target offers the same provider set.
    """
    provider_root = root / "target_providers" / target

    return {path.relative_to(provider_root) for path in provider_root.rglob("*.py")}
