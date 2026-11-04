#!/usr/bin/env bash
# A5 + CANN 9.1 accuracy suite. Per-case final states are in summary.txt/tsv/json.
# Usage: bash tests/python/operators/test_pr939_a5_cann91.sh [/path/to/repo]
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${1:-$PWD}"
if [[ $# -gt 0 ]]; then
  shift
fi
exec "${PYTHON_BIN:-python3}" "$script_dir/run_pr939_a5_cann91.py" "$repo_dir" \
  --device "${NPU_DEVICE:-0}" --timeout "${CASE_TIMEOUT:-900}" "$@"
