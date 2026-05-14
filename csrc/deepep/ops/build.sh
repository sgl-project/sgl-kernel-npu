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

# ASCEND910C (A3) series
# dependency: catlass
git config --add safe.directory "$script_path"
CATLASS_PATH=${CATLASS_DIR}/include
if [[ ! -d "${CATLASS_PATH}" ]]; then
    echo "dependency catlass is missing, try to fetch it..."
    if ! git clone -b catlass-v1-stable https://gitcode.com/cann/catlass.git "${CATLASS_DIR}"; then
        echo "catlass fetch failed"
        exit 1
    fi
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
