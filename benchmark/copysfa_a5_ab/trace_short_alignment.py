#!/usr/bin/env python3
"""One A5 short-sequence native-versus-offload case; no matrix or model load.

Defaults: 16 visible tokens, Q4, batch 1, 128 heads, C8192, NPU 0.
Use --query-width 1 for ordinary non-speculative decode.
All detailed diagnostics go into the requested output directory.
"""

from trace_alignment import main

if __name__ == "__main__":
    raise SystemExit(
        main(defaults={"context": 16, "batch": 1, "heads": 128}, short_only=True)
    )
