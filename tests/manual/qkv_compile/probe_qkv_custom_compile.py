"""NPU-only offline-kernel probe; no service startup. Accepts explicitly identified captured or synthetic kwargs."""

import argparse, hashlib, importlib.util, json
from pathlib import Path
import torch
import torch_npu  # noqa: F401 - initializes the NPU extension


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument(
        "--inputs",
        required=True,
        help="torch.save list of dictionaries matching the function signature; captured or labelled synthetic, at least two shapes",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--input-kind", choices=["captured", "synthetic"], required=True)
    p.add_argument(
        "--input-note",
        default="",
        help="For synthetic inputs record seed, dimensions and generator identity",
    )
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(a.device)
    original = load(a.original, "qkv_original_probe").split_qkv_rmsnorm_rope
    module = load(a.candidate, "qkv_custom_probe")
    op = module.split_qkv_rmsnorm_rope
    compiled = torch.compile(op, fullgraph=True, mode="max-autotune-no-cudagraphs")
    cases = torch.load(a.inputs, map_location="cpu", weights_only=True)
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("Supply at least two argument dictionaries")
    report = {
        "schema": str(op._schema),
        "device": a.device,
        "torch": torch.__version__,
        "input_kind": a.input_kind,
        "input_note": a.input_note,
        "sources": {
            label: {
                "path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for label, path in [
                ("original", a.original),
                ("candidate", a.candidate),
                ("inputs", a.inputs),
            ]
        },
        "cases": [],
    }
    (out / "identity.json").write_text(json.dumps(report, indent=2))
    try:
        for index, case in enumerate(cases):
            kw = {
                k: v.to(a.device) if isinstance(v, torch.Tensor) else v
                for k, v in case.items()
            }
            before = {
                k: v.clone() for k, v in kw.items() if isinstance(v, torch.Tensor)
            }
            ref = original(**kw)
            eager = op(**kw)
            actual = compiled(**kw)
            repeat = compiled(**kw)
            torch.npu.synchronize()
            differences = []
            for name, values in [
                ("custom_eager", eager),
                ("compiled", actual),
                ("compiled_repeat", repeat),
            ]:
                for j, (expected, value) in enumerate(zip(ref, values)):
                    differences.append(
                        {
                            "mode": name,
                            "output": j,
                            "equal": torch.equal(expected, value),
                            "maxdiff": float(
                                (expected.float() - value.float()).abs().max().item()
                            ),
                        }
                    )
            unchanged = all(torch.equal(kw[k], v) for k, v in before.items())
            # Explicit schema/fake/functionalization validation, not a numeric accuracy oracle.
            checked = torch.library.opcheck(
                op,
                kwargs=kw,
                args=(),
                test_utils=(
                    "test_schema",
                    "test_faketensor",
                    "test_aot_dispatch_dynamic",
                ),
            )
            entry = {
                "index": index,
                "shape": list(kw["input"].shape),
                "input_unchanged": unchanged,
                "comparisons": differences,
                "opcheck": checked,
            }
            report["cases"].append(entry)
            torch.save(
                {
                    "reference": [x.cpu() for x in ref],
                    "compiled": [x.cpu() for x in actual],
                },
                out / f"case_{index}.pt",
            )
            (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
            if not unchanged or not all(row["equal"] for row in differences):
                raise AssertionError(
                    "Same-kernel eager/compiled parity or nonmutation failed; inspect saved results"
                )
        if len({tuple(case["input"].shape) for case in cases}) < 2:
            raise ValueError("Probe requires two distinct input shapes")
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        raise
    print(
        "All same-kernel comparisons and registration checks passed; graph replay still unverified"
    )


if __name__ == "__main__":
    main()
