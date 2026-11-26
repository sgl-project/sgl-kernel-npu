"""Check the A3 and A5 DSV4.1 mHC build plans."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("arch", ["2201", "3510"])
def test_dsv41_mhc_pre_build_plan(arch):
    plan = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "setup.py",
                "--arch-version",
                arch,
                "--cann-version",
                "9.2.0",
                "--plan",
            ],
            cwd=ROOT,
        )
    )
    by_name = {item["name"]: item for item in plan}
    assert "mhc_pre_sinkhorn" in by_name["sgl_ops_transformer"]["ops"]
    assert "mhc_pre_sinkhorn" in by_name["sgl_ops_transformer_dsv4"]["ops"]
    experimental = by_name["sgl_ops_transformer_experimental"]
    assert experimental["ops"] == ["compressor_v2", "sgl_mhc_pre_sinkhorn"]
    assert experimental["commit"] == "53f5c56996be158b52579d9cdd484b54a38dfb64"
    assert "sgl_ops_transformer_mhc_v2" not in by_name
    if arch == "2201":
        assert "sgl_cann_recipes_infer" not in by_name
    else:
        assert experimental["soc"] == "ascend950"
        assert by_name["sgl_cann_recipes_infer"]["ops"] == {
            "kv_compress_epilog_v2": "kv_compress_epilog_v2"
        }
