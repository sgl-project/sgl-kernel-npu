"""Sweep request-union misses using the user's existing isolated wheel prefixes."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fixture import FixtureSpec
from run_matrix import load_config
from runtime_check import check_interpreter


def load_parameters(root):
    receipts = list(root.glob("unfixed-m512/run-receipt.json"))
    receipts += list(root.glob("retry-*/unfixed-m512/run-receipt.json"))
    if not receipts:
        raise ValueError("No previous unfixed-m512/run-receipt.json found")
    source = max(receipts, key=lambda path: path.stat().st_mtime)
    command = json.loads(source.read_text())["runs"][0]["command"]
    values = {}
    for name in (
        "batch",
        "heads",
        "index-heads",
        "capacity",
        "prefix",
        "context",
        "warmup",
        "rounds",
        "iterations",
    ):
        flag = "--" + name
        values[name] = int(command[command.index(flag) + 1])
    return source, values


def make_commands(root, output, parameters, misses, widths, modes, device, profile):
    for width in widths:
        for count in misses:
            FixtureSpec(
                width,
                parameters["batch"],
                parameters["heads"],
                parameters["index-heads"],
                parameters["capacity"],
                parameters["prefix"],
                parameters["context"],
                miss_counts=(count,),
                selection="disjoint",
            ).validate()
    commands = []
    for count in misses:
        for version in ("unfixed", "fixed"):
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_matrix.py")),
                "--arm-config",
                str(root / f"packages-{version}/arms.json"),
                "--output-root",
                str(output / f"{version}-m{count}"),
                "--device",
                str(device),
                "--miss-counts",
                str(count),
                "--selection",
                "disjoint",
                "--query-widths",
                *map(str, widths),
                "--modes",
                *modes,
            ]
            for name, value in parameters.items():
                command.extend(["--" + name, str(value)])
            if profile:
                command.append("--profile")
            commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--misses", type=int, nargs="+", default=[512, 513, 1024, 2048, 4096, 8192]
    )
    parser.add_argument(
        "--query-widths", type=int, choices=(1, 4), nargs="+", default=[4]
    )
    parser.add_argument(
        "--modes", choices=("eager", "graph"), nargs="+", default=["graph"]
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.misses)) != len(args.misses):
        parser.error("duplicate miss counts are not supported")
    source, parameters = load_parameters(args.run_root)
    output = args.output_root or args.run_root / (
        "more-misses-" + time.strftime("%Y%m%d-%H%M%S")
    )
    commands = make_commands(
        args.run_root,
        output,
        parameters,
        args.misses,
        args.query_widths,
        args.modes,
        args.device,
        args.profile,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "reference_receipt": str(source),
                    "parameters": parameters,
                    "selection": "disjoint",
                    "commands": commands,
                },
                indent=2,
            )
        )
        return 0
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    for version in ("unfixed", "fixed"):
        config = args.run_root / f"packages-{version}/arms.json"
        for executable in dict.fromkeys(a["python"] for a in load_config(config)):
            check_interpreter(executable)
    output.mkdir(parents=True)
    receipt = {
        "reference_receipt": str(source),
        "parameters": parameters,
        "selection": "disjoint",
        "runs": [],
    }
    print(f"Output: {output}\nParameters: {parameters}", flush=True)
    for command in commands:
        case = Path(command[command.index("--output-root") + 1]).name
        print(f"Starting {case}", flush=True)
        with (output / f"{case}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        receipt["runs"].append({"command": command, "exit_code": code})
        (output / "sweep-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if code:
            print(
                f"Stopped after {case}: exit {code}; log: {output / (case + '.log')}",
                flush=True,
            )
            return code
    return subprocess.run(
        [sys.executable, str(Path(__file__).with_name("compare.py")), str(output)]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
