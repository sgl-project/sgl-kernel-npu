"""Check that the A3 wheel preserves V1 and bundles a separate V2 operator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile", [None, "ops_transformer"])
def test_a3_mhc_pre_v1_and_v2_have_distinct_names(profile):
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
    main = next(item for item in plan if item["name"] == "sgl_ops_transformer")
    assert "mhc_pre_sinkhorn" in main["ops"]
    (vendor,) = [item for item in plan if "sgl_mhc_pre_sinkhorn" in item["ops"]]
    assert vendor["name"] == "sgl_ops_transformer_mhc_v2"
    assert vendor["commit"] == "53f5c56996be158b52579d9cdd484b54a38dfb64"
    assert "--experimental" not in vendor["commands"][0]
