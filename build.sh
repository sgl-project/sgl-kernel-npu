#!/bin/bash
set -e


BUILD_TYPE=0

while getopts ":b:" opt; do
    case ${opt} in
        b )
            # build_type 0: default mode, build all
            # build_type 1: only build deep_ep
            BUILD_TYPE=$OPTARG
            ;;
        \? )
            echo "unknown flag: -$OPTARG" 1>&2
            exit 1
            ;;
        : )
            echo "-$OPTARG requires a value" 1>&2
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))

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

    cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" -DASCEND_HOME_PATH=$ASCEND_HOME_PATH -DSOC_VERSION=$SOC_VERSION -DBUILD_TYPE=$BUILD_TYPE -B "$BUILD_DIR" -S .
    cmake --build "$BUILD_DIR" -j8 && cmake --build "$BUILD_DIR" --target install
    cd -
}

function build_deepep_kernels()
{
    KERNEL_DIR="csrc/deepep/ops"
    CUSTOM_OPP_DIR="${CURRENT_DIR}/python/deep_ep/deep_ep"

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
    ./build_out/custom_opp_*.run --install-path=$CUSTOM_OPP_DIR
    cd -
}

function make_deepep_package()
{
    cd python/deep_ep || exit

    cp -v ${OUTPUT_DIR}/lib/* "$CURRENT_DIR"/python/deep_ep/deep_ep/
    rm -rf "$CURRENT_DIR"/python/deep_ep/build
    python3 setup.py clean --all
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
    build_deepep_kernels

    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel
    fi
    make_deepep_package
    make_sgl_kernel_npu_package
}

main
