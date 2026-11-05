"""Runtime CANN SDK version detection, independent of the build tools."""

import os
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def detect_cann_version():
    """Read and cache the active SDK release; initialize CANN before first use."""
    sdk = os.environ.get("ASCEND_HOME_PATH")
    if not sdk:
        raise RuntimeError("Initialize CANN: ASCEND_HOME_PATH is missing")
    for path in (Path(sdk) / "version.cfg", Path(sdk) / "compiler/version.info"):
        if path.is_file():
            match = re.search(
                r"(?im)^\s*(?:version|version_dir)\s*=\s*(\d+\.\d+\.\d+)\s*$",
                path.read_text(),
            )
            if match:
                return match[1]
    raise RuntimeError("Cannot read an exact CANN release from SDK version metadata")
