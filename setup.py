#!/usr/bin/env python3
"""Build the REFACTOR distribution; build.sh also schedules independent packages."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "python/sgl_kernel_npu"))

if __name__ == "__main__":
    if os.environ.get("SGL_REFACTOR_STAGED_PACKAGE"):
        from build_tools.torch_wheel import setup_refactor
        setup_refactor()
    else:
        from build_tools.refactor import main
        try:
            main()
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            sys.exit(1)
