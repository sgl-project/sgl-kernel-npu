#!/bin/bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ATTENTIONS_BUILD_DIR="$PROJECT_ROOT/csrc/attentions/build"
readonly PACKAGE_DIR="$PROJECT_ROOT/python/sgl_kernel_npu/sgl_kernel_npu"
readonly COMPUTE_UNIT="${1:-ascend910b}"

rm -rf "$ATTENTIONS_BUILD_DIR/output" "$ATTENTIONS_BUILD_DIR/vendors"

"$ATTENTIONS_BUILD_DIR/build_ascendc_ops.sh" \
    -n "hc_pre;hc_post" \
    -c "$COMPUTE_UNIT"

rm -rf "$PACKAGE_DIR/vendors"
mkdir -p "$PACKAGE_DIR/vendors"
cp -a "$ATTENTIONS_BUILD_DIR/vendors/." "$PACKAGE_DIR/vendors/"

echo "Bundled mHC custom operators into $PACKAGE_DIR/vendors"
