"""Resolve the installed native operator; never silently change backends."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _operator():
    import torch

    from ....operators.utils.generation import get_hardware

    manifest_path = Path(__file__).resolve().parents[3] / "_build_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if "apply_token_bitmask" not in manifest["operators"]:
            raise RuntimeError(
                "This wheel does not include ascendc apply_token_bitmask; rebuild with --ops apply_token_bitmask or --ops all"
            )
        if int(manifest["arch_version"]) != get_hardware().arch_id:
            raise RuntimeError(
                "The wheel ARCH does not match this NPU; install a wheel built for its --arch-version"
            )
    try:
        return torch.ops.npu.apply_token_bitmask
    except AttributeError as error:
        raise RuntimeError(
            "The installed native library does not register npu::apply_token_bitmask"
        ) from error


def apply_token_bitmask(logits, bitmask, indices=None):
    return _operator()(logits, bitmask, indices)
