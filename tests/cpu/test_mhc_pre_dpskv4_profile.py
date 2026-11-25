"""Check A3 and A5 HcPre build plans."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("arch", ["2201", "3510"])
def test_hc_pre_build_plan(arch):
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
    recipes = next(item for item in plan if item["name"] == "sgl_cann_recipes_infer")
    assert recipes["ops"]["npu_hc_pre_v2"] == "hc_pre"
    assert ("vllm_hc_pre" in recipes) == (arch == "2201")
    if arch == "2201":
        assert (
            recipes["vllm_hc_pre"]["commit"]
            == "2b1a0514511b67b751580cef0fa11c9b053bce65"
        )
