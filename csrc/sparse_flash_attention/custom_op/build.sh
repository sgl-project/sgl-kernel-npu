#!/bin/bash
set -euo pipefail

readonly SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOC_VERSION="${1:?SOC_VERSION is required}"
readonly INSTALL_DIR="$(realpath -m "${2:?INSTALL_DIR is required}")"
readonly PROJECT_DIR="${SOURCE_DIR}/build_project"
readonly BUILD_DIR="${PROJECT_DIR}/build_out"
readonly VENDOR_NAME="sgl_kernel_npu"

export OPS_PROJECT_NAME=aclnnInner

case "$SOC_VERSION" in
    Ascend910B1 | ascend910b*)
        COMPUTE_UNIT="ascend910b"
        MSOPGEN_SOC="Ascend910B1"
        ;;
    Ascend910_93* | ascend910_93*)
        COMPUTE_UNIT="ascend910_93"
        MSOPGEN_SOC="Ascend910_9382"
        ;;
    Ascend950* | ascend950*)
        COMPUTE_UNIT="ascend950"
        MSOPGEN_SOC="Ascend950PR_9599"
        ;;
    *)
        echo "Unsupported SparseFlashAttention SOC_VERSION: $SOC_VERSION" >&2
        exit 1
        ;;
esac

rm -rf -- "$PROJECT_DIR"
msopgen gen \
    -i "${SOURCE_DIR}/Scaffold.json" \
    -c "ai_core-${MSOPGEN_SOC}" \
    -f pytorch \
    -lan cpp \
    -out "$PROJECT_DIR"

rm -rf -- "${PROJECT_DIR}/op_host" "${PROJECT_DIR}/op_kernel"
cp -a "${SOURCE_DIR}/CMakeLists.txt" "${PROJECT_DIR}/CMakeLists.txt"
cp -a "${SOURCE_DIR}/op_host" "${PROJECT_DIR}/op_host"
cp -a "${SOURCE_DIR}/op_kernel" "${PROJECT_DIR}/op_kernel"
cp -a "${SOURCE_DIR}/common" "${PROJECT_DIR}/common"

cmake \
    -S "$PROJECT_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_SOURCE_PACKAGE=OFF \
    -DENABLE_BINARY_PACKAGE=ON \
    -DASCEND_COMPUTE_UNIT="$COMPUTE_UNIT" \
    -Dvendor_name="$VENDOR_NAME" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is required}" \
    -DASCEND_PYTHON_EXECUTABLE=python3 \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR"

cmake --build "$BUILD_DIR" --target binary -j "${MAX_JOBS:-16}"
cmake --build "$BUILD_DIR" --target package -j "${MAX_JOBS:-16}"

installer="$(find "$BUILD_DIR" -maxdepth 1 -type f -name 'custom_opp*.run' -print -quit)"
if [[ -z "$installer" ]]; then
    echo "SparseFlashAttention custom OPP installer was not generated" >&2
    exit 1
fi

rm -rf -- "${INSTALL_DIR}/vendors/${VENDOR_NAME}"
chmod +x "$installer"
"$installer" --install-path="$INSTALL_DIR"
