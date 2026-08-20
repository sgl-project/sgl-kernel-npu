#!/bin/bash

export OPS_PROJECT_NAME=aclnnInner

if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    else
        export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath $(dirname $0))

mkdir -p "${script_path}/third_party"
CATLASS_DIR="${script_path}/third_party/catlass"
CATLASS_REPO_URL="https://gitcode.com/cann/catlass.git"
CATLASS_REF_A3="catlass-v1-stable"
CATLASS_REF_A5="v1.6.1"

# ASCEND910C (A3) and ASCEND950 (A5) series
# shared dependency: catlass
git config --add safe.directory "$script_path"
CATLASS_PATH=${CATLASS_DIR}/include

CATLASS_REF="${CATLASS_REF_A3}"
if [[ "${ASCEND_COMPUTE_UNIT}" =~ ^ascend950 ]]; then
    CATLASS_REF="${CATLASS_REF_A5}"
fi

echo "using ASCEND_COMPUTE_UNIT: ${ASCEND_COMPUTE_UNIT:-<unset>}"
echo "using catlass ref: ${CATLASS_REF}"

if [[ ! -d "${CATLASS_DIR}/.git" ]]; then
    echo "dependency catlass is missing, try to fetch it..."
    if ! git clone "${CATLASS_REPO_URL}" "${CATLASS_DIR}"; then
        echo "catlass fetch failed"
        exit 1
    fi
fi

catlass_need_update="1"
if git -C "${CATLASS_DIR}" rev-parse -q --verify "${CATLASS_REF}^{commit}" >/dev/null 2>&1; then
    current_catlass_commit=$(git -C "${CATLASS_DIR}" rev-parse HEAD)
    target_catlass_commit=$(git -C "${CATLASS_DIR}" rev-parse "${CATLASS_REF}^{commit}")
    if [[ "${current_catlass_commit}" == "${target_catlass_commit}" ]]; then
        catlass_need_update="0"
        echo "catlass is already at target ref: ${CATLASS_REF}"
    fi
fi

if [[ "${catlass_need_update}" == "1" ]]; then
    if ! git -C "${CATLASS_DIR}" fetch --tags origin; then
        echo "catlass fetch tags failed"
        exit 1
    fi

    if ! git -C "${CATLASS_DIR}" checkout -f "${CATLASS_REF}"; then
        echo "catlass checkout ${CATLASS_REF} failed"
        exit 1
    fi
fi

if [[ ! -d "${CATLASS_PATH}" ]]; then
    echo "catlass include path missing after checkout: ${CATLASS_PATH}"
    exit 1
fi
# dependency: cann-toolkit file moe_distribute_base.h
HCCL_STRUCT_FILE_PATH=$(find -L "${ASCEND_HOME_PATH}" -name "moe_distribute_base.h" 2>/dev/null | head -n1)
if [ -z "$HCCL_STRUCT_FILE_PATH" ]; then
    echo "cannot find moe_distribute_base.h file in CANN env"
    exit 1
fi
# for dispatch & combine..
# cp -vf "$HCCL_STRUCT_FILE_PATH" "$script_path/op_kernel/"

# for dispatch_ffn_combine & dispatch_ffn_combine_bf16
TARGET_DIR="$script_path/op_kernel/dispatch_ffn_combine_kernel/utils/"
TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"
# TARGET_DIR_BF16="$script_path/op_kernel/dispatch_ffn_combine_bf16_kernel/utils/"
# TARGET_FILE_BF16="$TARGET_DIR_BF16/$(basename "$HCCL_STRUCT_FILE_PATH")"
echo "*************************************"
echo $HCCL_STRUCT_FILE_PATH
echo "$TARGET_DIR"
# cp -v "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
# cp -v "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR_BF16"
sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"
# sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE_BF16"
# sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE_BF16"

BUILD_DIR="build_out"
HOST_NATIVE_DIR="host_native_tiling"
mkdir -p build_out
rm -rf build_out/*

ENABLE_CROSS="-DENABLE_CROSS_COMPILE=True"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
ENABLE_LIBRARY="-DASCEND_PACK_SHARED_LIBRARY=True"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake -S . -B "$BUILD_DIR" --preset=default
cmake --build "$BUILD_DIR" --target binary -j$(nproc)
cmake --build "$BUILD_DIR" --target $target -j$(nproc)
