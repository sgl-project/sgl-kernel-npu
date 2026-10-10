#!/bin/bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MHC_SOURCE_DIR="${PROJECT_ROOT}/csrc/mhc"
readonly OUTPUT_DIR="${PROJECT_ROOT}/output"
readonly COMPUTE_UNIT="${1:-ascend910b}"

case "${COMPUTE_UNIT}" in
    ascend910b) MSOPGEN_COMPUTE_UNIT="Ascend910B1" ;;
    ascend910_93) MSOPGEN_COMPUTE_UNIT="Ascend910_9382" ;;
    *) MSOPGEN_COMPUTE_UNIT="${COMPUTE_UNIT}" ;;
esac
readonly MSOPGEN_COMPUTE_UNIT

if ! command -v msopgen >/dev/null 2>&1; then
    echo "msopgen is required to build the mHC custom operators" >&2
    exit 1
fi

build_root="$(mktemp -d /tmp/sgl-kernel-npu-mhc.XXXXXX)"
trap 'rm -rf "${build_root}"' EXIT
opp_project="${build_root}/ops"

# Generate the CANN custom-OPP project at build time. Only the small MHC-specific
# CMake files and the official operator sources are kept in this repository.
msopgen gen \
    -i "${MHC_SOURCE_DIR}/AddCustom.json" \
    -c "ai_core-${MSOPGEN_COMPUTE_UNIT}" \
    -f pytorch \
    -lan cpp \
    -out "${opp_project}"

rm -f "${opp_project}/op_host/add_custom"* \
      "${opp_project}/op_kernel/add_custom"*

cp -a "${MHC_SOURCE_DIR}/hc_pre/op_host/." "${opp_project}/op_host/"
cp -a "${MHC_SOURCE_DIR}/hc_post/op_host/." "${opp_project}/op_host/"
cp -a "${MHC_SOURCE_DIR}/hc_pre/op_kernel/." "${opp_project}/op_kernel/"
cp -a "${MHC_SOURCE_DIR}/hc_post/op_kernel/." "${opp_project}/op_kernel/"
cp -a "${MHC_SOURCE_DIR}/ops/include" "${opp_project}/op_host/"
cp -f "${MHC_SOURCE_DIR}/ops/op_host/CMakeLists.txt" \
    "${opp_project}/op_host/CMakeLists.txt"
cp -f "${MHC_SOURCE_DIR}/ops/op_kernel/CMakeLists.txt" \
    "${opp_project}/op_kernel/CMakeLists.txt"

(
    cd "${opp_project}"
    chmod +x build.sh
    ./build.sh
)

run_package="$(find "${opp_project}/build_out" \
    -maxdepth 1 \
    -type f \
    -name 'custom_opp*.run' \
    -print \
    -quit)"
if [[ -z "${run_package}" ]]; then
    echo "Cannot find the generated mHC custom-OPP package" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
destination="${OUTPUT_DIR}/sgl_kernel_npu_mhc_ops-${COMPUTE_UNIT}-linux.$(uname -m).run"
cp -v "${run_package}" "${destination}"
echo "mHC custom-op package: ${destination}"
