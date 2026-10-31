#!/usr/bin/env bash
# CPU-only serialization parity against the original CANN tiling classes.
set -eo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 ORIGINAL_SOURCE_ROOT NEW_OUTPUT_DIR" >&2; exit 2; }
source_root=$(cd "$1" && pwd)
output=$2
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
mkdir -p "$output"
output=$(cd "$output" && pwd)
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
source "${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
trap 'rc=$?; echo "$rc" > "$output/exitcode"' EXIT
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$ASCEND_HOME_PATH/include" -I"$source_root/csrc" \
  "$repo/tests/test_copy_sfa_c8_tiling_layout.cpp" \
  -L"$ASCEND_HOME_PATH/lib64" -lregister -o "$output/check" \
  > "$output/compile.log" 2>&1
"$output/check" > "$output/layout.log" 2>&1
cat "$output/layout.log"
