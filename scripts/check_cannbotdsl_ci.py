"""Validate the A5 CI environment, installed sources and executed test coverage."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu"
REQUIRED_CASES = (
    "test_indexer_prologue_k_npu[mode0_with_scale]",
    "test_indexer_prologue_k_npu[mode1_g8]",
    "test_quant_lightning_indexer[pa128_ragged]",
    "test_quant_lightning_indexer[pa64_causal_candidates]",
    "test_quant_lightning_indexer[tnd_causal]",
    "test_quant_sparse_lightning_indexer[",
    "test_mixed_quant_sparse_flash_mla[mixed_decode]",
    "test_mixed_quant_sparse_flash_mla[ori_only]",
    "test_metadata_replay_reads_updated_lengths[mqsmla]",
    "test_metadata_replay_reads_updated_lengths[qli]",
    "test_metadata_replay_reads_updated_lengths[qsli]",
)


def check_junit(path):
    cases = ET.parse(path).getroot().findall(".//testcase")
    unsuccessful = [
        c.get("name")
        for c in cases
        if any(c.find(tag) is not None for tag in ("skipped", "failure", "error"))
    ]
    passed = [c.get("name", "") for c in cases if c.get("name") not in unsuccessful]
    missing = [
        prefix
        for prefix in REQUIRED_CASES
        if not any(n.startswith(prefix) for n in passed)
    ]
    if unsuccessful or missing:
        raise RuntimeError(
            f"Incomplete device validation: unsuccessful={unsuccessful}, missing={missing}"
        )
    return {"passed": len(passed), "skipped": 0, "required_cases": list(REQUIRED_CASES)}


def environment(expected_commit):
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if actual != expected_commit:
        raise RuntimeError(f"Source commit mismatch: {actual} != {expected_commit}")
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError("This CI contract requires Linux amd64 and Python 3.12")
    sys.path.insert(0, str(ROOT / "python/sgl_kernel_npu"))
    from build_tools.third_party import detect_cann_version

    cann = detect_cann_version()
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torch_npu", "cannbot-dsl")
    }
    if cann != "9.2.0" or versions["cannbot-dsl"] != "0.8.0":
        raise RuntimeError(
            f"Expected CANN 9.2.0 / DSL 0.8.0; found {cann} / {versions}"
        )
    if any(versions[name].split("+")[0] != "2.13.0" for name in ("torch", "torch_npu")):
        raise RuntimeError(f"Expected torch/torch_npu 2.13.0; found {versions}")
    # Avoid importing sgl_kernel_npu before its wheel exists.
    sys.path.pop(0)
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; skipping device tests is not a CI pass")
    torch.npu.set_device(0)
    path = SOURCE / "operators/utils/generation.py"
    spec = importlib.util.spec_from_file_location("_ci_generation", path)
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
    hardware = generation.get_hardware()
    if hardware.arch_id != 3510:
        raise RuntimeError(f"Expected A5 / ARCH 3510, found {hardware}")
    return dict(
        commit=actual,
        cann=cann,
        versions=versions,
        python=sys.version,
        arch=hardware.arch_id,
        device_name=torch.npu.get_device_name(0),
        visible_devices=torch.npu.device_count(),
    )


def installed_sources():
    import sgl_kernel_npu

    package = Path(sgl_kernel_npu.__file__).resolve().parent
    if package == SOURCE.resolve():
        raise RuntimeError("Expected an installed wheel, not the source checkout")
    files = list((SOURCE / "implementation/cannbot-dsl").rglob("*.py"))
    files += [SOURCE / "implementation/cannbot-dsl/UPSTREAM.json"]
    upstream = json.loads(files[-1].read_text())
    for item in upstream["files"]:
        domain = "attention" if item["file"].startswith("mixed_") else "indexer"
        files.append(SOURCE / "operators" / domain / item["file"])
    files += [
        SOURCE / "operators" / d / "__init__.py" for d in ("attention", "indexer")
    ]
    hashes = {}
    for source in files:
        relative = source.relative_to(SOURCE)
        target = package / relative
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Installed source differs from checkout: {target}")
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = json.loads((package / "_build_manifest.json").read_text())
    if manifest["arch_version"] != "3510":
        raise RuntimeError("Installed wheel is not an A5 build")
    staged = ROOT / "build/torch/package/sgl_kernel_npu"
    for relative in ("_build_manifest.json", "lib/libsgl_kernel_npu.so"):
        if (staged / relative).read_bytes() != (package / relative).read_bytes():
            raise RuntimeError(
                f"Installed artifact differs from this build: {relative}"
            )
    return dict(package=str(package), manifest=manifest, source_hashes=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("environment", "installed", "results"))
    parser.add_argument("--commit")
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "environment":
        if not args.commit:
            parser.error("environment requires --commit")
        result = environment(args.commit)
    elif args.phase == "installed":
        result = installed_sources()
    else:
        if args.junit is None:
            parser.error("results requires --junit")
        result = check_junit(args.junit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
