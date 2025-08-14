#!/bin/bash
set -e

SOC_VERSION="${1:-Ascend910_9382}"

if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "ascend path: ${ASCEND_HOME_PATH}"
source $(dirname ${ASCEND_HOME_PATH})/set_env.sh

CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
VERSION="1.0.0"
OUTPUT_DIR=$CURRENT_DIR/output
echo "outpath: ${OUTPUT_DIR}"

COMPILE_OPTIONS=""

function build_kernels()
{
    CMAKE_DIR=""
    BUILD_DIR="build"

    cd "$CMAKE_DIR" || exit

    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR

    cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" -DASCEND_HOME_PATH=$ASCEND_HOME_PATH -DSOC_VERSION=$SOC_VERSION -B "$BUILD_DIR" -S .
    cmake --build "$BUILD_DIR" -j8 && cmake --build "$BUILD_DIR" --target install
    cd -
}

function make_deepep_package()
{
    cd python/deep_ep || exit

    cp -v ${OUTPUT_DIR}/lib/* "$CURRENT_DIR"/python/deep_ep/deep_ep/
    rm -rf "$CURRENT_DIR"/python/deep_ep/dist
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/python/deep_ep/dist/deep_ep*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/deep_ep/dist
    cd -
}

function make_sgl_kernel_npu_package()
{
    cd python/sgl_kernel_npu || exit

    rm -rf "$CURRENT_DIR"/python/sgl_kernel_npu/dist
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/python/sgl_kernel_npu/dist/sgl_kernel_npu*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/sgl_kernel_npu/dist
    cd -
}

function main()
{
    build_kernels

    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel
    fi
    make_deepep_package
    make_sgl_kernel_npu_package
}

main