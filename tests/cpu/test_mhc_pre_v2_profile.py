"""Check which CANN vendor provides MhcPreSinkhorn on A3."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile", [None, "ops_transformer", "ops_transformer_dsv4"])
def test_a3_mhc_pre_v2_has_one_vendor(profile):
    command = [
        sys.executable,
        "setup.py",
        "--arch-version",
        "2201",
        "--cann-version",
        "9.2.0",
        "--plan",
    ]
    if profile is not None:
        command.extend(["--third-party", profile])
    plan = json.loads(subprocess.check_output(command, cwd=ROOT))
    (vendor,) = [item for item in plan if "mhc_pre_sinkhorn" in item["ops"]]
    assert vendor["name"] == "sgl_ops_transformer_mhc"
    assert vendor["commit"] == "53f5c56996be158b52579d9cdd484b54a38dfb64"
    assert "--experimental" not in vendor["commands"][0]
