#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

PROJECT_ROOT=$(dirname $( dirname $(dirname "$CURRENT_DIR")))

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"
SOC_VERSION="Ascend910_9382"
RUN_MODE=npu


if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

if [ -n "$ASCEND_INCLUDE_DIR" ]; then
    ASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR
else
    ASCEND_INCLUDE_DIR=${_ASCEND_INSTALL_PATH}/include
fi

if [ -n "$SHMEM_HOME_PATH" ]; then
    _SHMEM_HOME_PATH=$SHMEM_HOME_PATH
else
    _SHMEM_HOME_PATH=/usr/local/Ascend/shmem/latest
fi

ZCCL_HOME_PATH=zccl
ZCCL_INCLUDE_DIR=${ZCCL_HOME_PATH}/include
ZCCL_LIB_DIR=${ZCCL_HOME_PATH}/lib

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "[INFO]: Current compile soc version is ${SOC_VERSION}"

set -e
rm -rf build out
mkdir -p build
cmake -B build \
    -DSHMEM_HOME_PATH=${_SHMEM_HOME_PATH} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH} \
    -DASCEND_HOME_PATH=${ASCEND_HOME_PATH} \
    -DASCEND_INCLUDE_DIR=${ASCEND_INCLUDE_DIR} \
    -DZCCL_INCLUDE_DIR=${ZCCL_INCLUDE_DIR} \
    -DZCCL_LIB_DIR=${ZCCL_LIB_DIR}
cmake --build build -j
cmake --install build

