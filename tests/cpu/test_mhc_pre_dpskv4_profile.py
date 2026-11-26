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
    main = next(item for item in plan if item["name"] == "sgl_ops_transformer")
    assert "mhc_pre_sinkhorn" in main["ops"]
    by_name = {item["name"]: item for item in plan}
    v2 = by_name["sgl_ops_transformer_mhc_v2"]
    assert v2["ops"] == ["sgl_mhc_pre_sinkhorn"]
    assert v2["commit"] == "53f5c56996be158b52579d9cdd484b54a38dfb64"
    if arch == "2201":
        assert "sgl_cann_recipes_infer" not in by_name
    else:
        assert v2["soc"] == "ascend950"
        assert by_name["sgl_cann_recipes_infer"]["ops"] == {
            "kv_compress_epilog_v2": "kv_compress_epilog_v2"
        }
