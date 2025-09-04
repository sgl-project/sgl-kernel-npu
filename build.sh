#!/bin/bash
set -e

BUILD_DEEPEP_MODULE="ON"
BUILD_KERNELS_MODULE="ON"
BUILD_MEMORY_SAVER_MODULE="ON"

ONLY_BUILD_DEEPEP_ADAPTER_MODULE="OFF"
ONLY_BUILD_DEEPEP_KERNELs_MODULE="OFF"
ONLY_BUILD_MEMORY_SAVER_MODULE="OFF"

while getopts ":a:h" opt; do
    case ${opt} in
        a )
            BUILD_DEEPEP_MODULE="OFF"
            BUILD_KERNELS_MODULE="OFF"
            BUILD_MEMORY_SAVER_MODULE="OFF"
            case "$OPTARG" in
                deepep )
                    BUILD_DEEPEP_MODULE="ON"
                    ;;
                kernels )
                    BUILD_KERNELS_MODULE="ON"
                    ;;
                deepep-adapter )
                    BUILD_DEEPEP_MODULE="ON"
                    ONLY_BUILD_DEEPEP_ADAPTER_MODULE="ON"
                    ;;
                deepep-kernels )
                    BUILD_DEEPEP_MODULE="ON"
                    ONLY_BUILD_DEEPEP_KERNELs_MODULE="ON"
                    ;;
                memory-saver )
                    BUILD_MEMORY_SAVER_MODULE="ON"
                    ONLY_BUILD_MEMORY_SAVER_MODULE="ON"
                    ;;
                * )
                    echo "Error: Invalid Value"
                    echo "Allowed value: deepep|kernels|deepep-adapter|deepep-kernels|memory-saver"
                    exit 1
                    ;;
            esac
            ;;
        h )
            echo "Use ./build.sh build all modules.\n"
            echo "Use './build.sh -a <target>' to build specific parts of the project."
            echo "    <target> can be: all, deepep, kernels, deepep-adapter, deepep-kernels"
            echo "    deepep            Only build deep_ep."
            echo "    kernels           Only build sgl_kernel_npu."
            echo "    deepep-adapter    Only build deepep adapter layer and use old build of deepep kernels."
            echo "    deepep-kernels    Only build deepep kernels and use old build of deepep adapter layer."
            echo "    memory-saver      Only build torch_memory_saver (under contrib)."
            exit 1
            ;;
        \? )
            echo "Error: unknown flag: -$OPTARG" 1>&2
            exit 1
            ;;
        : )
            echo "Error: -$OPTARG requires a value" 1>&2
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
mkdir -p $OUTPUT_DIR
echo "outpath: ${OUTPUT_DIR}"

COMPILE_OPTIONS=""

function build_kernels()
{
    if [[ "$ONLY_BUILD_DEEPEP_KERNELs_MODULE" == "ON" ]]; then return 0; fi
    if [[ "$BUILD_KERNELS_MODULE" != "ON" ]]; then return 0; fi

    CMAKE_DIR=""
    BUILD_DIR="build"

    cd "$CMAKE_DIR" || exit

    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR

    cmake $COMPILE_OPTIONS -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" -DASCEND_HOME_PATH=$ASCEND_HOME_PATH -DSOC_VERSION=$SOC_VERSION -DBUILD_DEEPEP_MODULE=$BUILD_DEEPEP_MODULE -DBUILD_KERNELS_MODULE=$BUILD_KERNELS_MODULE -B "$BUILD_DIR" -S .
    cmake --build "$BUILD_DIR" -j8 && cmake --build "$BUILD_DIR" --target install
    cd -
}

function build_deepep_kernels()
{
    if [[ "$ONLY_BUILD_DEEPEP_ADAPTER_MODULE" == "ON" ]]; then return 0; fi
    if [[ "$BUILD_DEEPEP_MODULE" != "ON" ]]; then return 0; fi

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

function build_memory_saver()
{
    if [[ "$BUILD_MEMORY_SAVER_MODULE" != "ON" ]]; then return 0; fi
    echo "[memory_saver] Building torch_memory_saver via setup.py"
    cd contrib/torch_memory_saver/python || exit
    rm -rf "$CURRENT_DIR"/contrib/torch_memory_saver/python/build
    rm -rf "$CURRENT_DIR"/contrib/torch_memory_saver/python/dist
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/contrib/torch_memory_saver/python/torch_memory_saver*.whl "${OUTPUT_DIR}/"
    rm -rf "$CURRENT_DIR"/contrib/torch_memory_saver/python/dist
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
    build_memory_saver
    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel
    fi
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" ]]; then
        make_deepep_package
    fi
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        make_sgl_kernel_npu_package
    fi

}

main
