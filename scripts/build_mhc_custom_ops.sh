#!/bin/bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MHC_PROJECT_DIR="${PROJECT_ROOT}/csrc/mhc/custom_ops"
readonly OUTPUT_DIR="${PROJECT_ROOT}/output"
readonly COMPUTE_UNIT="${1:-ascend910b}"

if [[ ! -x "${MHC_PROJECT_DIR}/build.sh" ]]; then
    echo "Missing ${MHC_PROJECT_DIR}/build.sh" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
(
    cd "${MHC_PROJECT_DIR}"
    OPS_CPU_NUMBER="${OPS_CPU_NUMBER:-16}" \
        ./build.sh \
        -n "hc_pre;hc_post" \
        -c "${COMPUTE_UNIT}" \
        --disable-check-compatible
)

run_package="$(find "${MHC_PROJECT_DIR}/output" \
    -maxdepth 1 \
    -type f \
    -name 'CANN-custom_ops-*.run' \
    -print \
    -quit)"
if [[ -z "${run_package}" ]]; then
    echo "Cannot find the generated mHC custom-op package" >&2
    exit 1
fi

destination="${OUTPUT_DIR}/sgl_kernel_npu_mhc_ops-${COMPUTE_UNIT}-linux.$(uname -m).run"
cp -v "${run_package}" "${destination}"
echo "mHC custom-op package: ${destination}"
