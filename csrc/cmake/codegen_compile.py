"""Forward SDK compiler invocations with the selected Host source's options."""

import json
import pathlib
import subprocess
import sys


def main():
    with open(sys.argv[1], encoding="utf-8") as stream:
        config = json.load(stream)
    arguments = sys.argv[2:]
    options = []
    # Link invocations must never inherit a Host translation unit's settings.
    if any(flag in arguments for flag in ("-c", "-E", "-S")):
        source_options = {
            str(pathlib.Path(source).resolve()): source_flags
            for source, source_flags in config["source_options"].items()
        }
        for argument in arguments:
            if not argument.startswith("-"):
                key = str(pathlib.Path(argument).resolve())
                if key in source_options:
                    options.extend(source_options[key])
    return subprocess.run(config["compiler"] + arguments + options).returncode


if __name__ == "__main__":
    sys.exit(main())
