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