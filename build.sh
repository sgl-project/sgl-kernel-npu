#!/bin/bash
set -e

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

function build_deepep()
{
    CMAKE_DIR="csrc"
    BUILD_DIR="build"

    cd "$CMAKE_DIR" || exit

    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR

    cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" -DASCEND_HOME_PATH=$ASCEND_HOME_PATH -B "$BUILD_DIR" -S .
    cmake --build "$BUILD_DIR" -j8 && cmake --build "$BUILD_DIR" --target install
    cd -
}

function make_package()
{
    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel
    fi

    PYTHON_DIR="python"
    cd "$PYTHON_DIR" || exit

    cp -v ${OUTPUT_DIR}/lib/* "$CURRENT_DIR"/python/deep_ep/
    rm -rf "$CURRENT_DIR"/python/dist
    python3 setup.py build
    cd -
}

function build_kernels()
{
    KERNEL_DIR="csrc/kernels"

    CUSTOM_OPP_DIR="${CURRENT_DIR}/deep_ep"

    cd "$KERNEL_DIR" || exit

    chmod +x build.sh
    chmod +x cmake/util/gen_ops_filter.sh
    ./build.sh

    custom_opp_file=$(find ./build_out -maxdepth 1 -type f -name "custom_opp*.run")
    if [ -z "$custom_opp_file" ]; then
        echo "can not find run package"
        exit 1
    else
        echo "find run package: $custom_opp_file"
        chmod +x "$custom_opp_file"
    fi
    # echo "./build_out/custom_opp_*.run --install-path=$CUSTOM_OPP_DIR"
    ./build_out/custom_opp_*.run --install-path=$CUSTOM_OPP_DIR

    cd -
}


function main()
{
    build_deepep
    make_package
}

main