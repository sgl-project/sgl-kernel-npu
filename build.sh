#!/bin/bash
set -e

BUILD_ATTENTIONS_MODULE="ON"
BUILD_DEEPEP_MODULE="ON"
BUILD_DEEPEP_OPS="ON"
BUILD_KERNELS_MODULE="ON"
BUILD_MEMORY_SAVER_MODULE="ON"

ONLY_BUILD_DEEPEP_ADAPTER_MODULE="OFF"
ONLY_BUILD_DEEPEP_KERNELs_MODULE="OFF"
ONLY_BUILD_MEMORY_SAVER_MODULE="OFF"

DEBUG_MODE="OFF"

while getopts ":a:hd" opt; do
    case ${opt} in
        a )
            BUILD_DEEPEP_MODULE="OFF"
            BUILD_KERNELS_MODULE="OFF"
            BUILD_MEMORY_SAVER_MODULE="OFF"
            case "$OPTARG" in
                deepep )
                    BUILD_DEEPEP_MODULE="ON"
                    BUILD_DEEPEP_OPS="ON"
                    ;;
                deepep2 )
                    BUILD_DEEPEP_MODULE="ON"
                    BUILD_DEEPEP_OPS="OFF"
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
        d )
            DEBUG_MODE="ON"
            ;;
        h )
            echo "Use './build.sh' build all modules."
            echo "Use './build.sh -a <target>' to build specific parts of the project."
            echo "    <target> can be:"
            echo "    deepep            Only build deep_ep."
            echo "    kernels           Only build sgl_kernel_npu."
            echo "    deepep-adapter    Only build deepep adapter layer and use old build of deepep kernels."
            echo "    deepep-kernels    Only build deepep kernels and use old build of deepep adapter layer."
            echo "    memory-saver      Only build torch_memory_saver (under contrib)."
            exit 1
            ;;
        \? )
            echo "Error: unknown flag: -$OPTARG" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
        : )
            echo "Error: -$OPTARG requires a value" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))


export DEBUG_MODE=$DEBUG_MODE

# Chip mapping:
# - deepep  → A3+ (Ascend910_9382)
# - deepep2 → A2  (Ascend910B1)

if [[ "$BUILD_DEEPEP_OPS" == "ON" ]]; then
    SOC_VERSION="${1:-Ascend910_9382}"
else
    SOC_VERSION="${1:-Ascend910B1}"
fi

echo "Use SOC_VERSION: $SOC_VERSION"

echo "=== Fixing ASCConfig for CANN 8.3 / A2 ==="

# Prioritize using the ASCEND_HOME_PATH environment variable
# If empty or points to latest, automatically select the actual installation version
if [ -z "$ASCEND_HOME_PATH" ] || [[ "$ASCEND_HOME_PATH" == *"/latest" ]]; then
    REAL_ASCEND_PATH=$(ls -d /usr/local/Ascend/ascend-toolkit/* \
        | grep -v latest \
        | sort -V \
        | tail -1)

    if [ -n "$REAL_ASCEND_PATH" ]; then
        export ASCEND_HOME_PATH="$REAL_ASCEND_PATH"
    else
        echo "Error: Cannot find Ascend toolkit installation"
        exit 1
    fi
fi

echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

# 主动查找并设置 ASCConfig.cmake 路径
ASC_CONFIG_CMAKE=$(find "$ASCEND_HOME_PATH" -name "ASCConfig.cmake" -type f 2>/dev/null | head -n1)
if [ -n "$ASC_CONFIG_CMAKE" ]; then
    ASC_CMAKE_DIR=$(dirname "$ASC_CONFIG_CMAKE")
    echo "Found ASCConfig.cmake at: $ASC_CONFIG_CMAKE"
    export CMAKE_PREFIX_PATH="$ASC_CMAKE_DIR:$CMAKE_PREFIX_PATH"
    export ASC_DIR="$ASC_CMAKE_DIR"
    echo "Set CMAKE_PREFIX_PATH and ASC_DIR successfully"
else
    echo "Warning: Cannot find ASCConfig.cmake"
fi

# Get Current CANN Toolkit Installation Path
if [ -n "$ASCEND_HOME_PATH" ]; then
    _CANN_TOOLKIT_INSTALL_PATH="$ASCEND_HOME_PATH"
else
    _CANN_TOOLKIT_INSTALL_PATH=$(cat /etc/Ascend/ascend_cann_install.info | grep "Toolkit_InstallPath" | awk -F'=' '{print $2}')
fi
source ${_CANN_TOOLKIT_INSTALL_PATH}/set_env.sh
echo -e "\e[1;32mDetected CANN Toolkit Installation Path: ${_CANN_TOOLKIT_INSTALL_PATH}\e[0m"
echo -e "\e[1;33mDouble Checking Environment Variables:\e[0m"
echo -e "\e[1;32mASCEND_HOME_PATH: ${ASCEND_HOME_PATH}\e[0m"
echo -e "\e[1;32mASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}\e[0m"

ASCEND_INCLUDE_DIR=${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include
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
    if [[ "$ONLY_BUILD_MEMORY_SAVER_MODULE" == "ON" ]]; then return 0; fi

    CMAKE_DIR=""
    BUILD_DIR="build"

    cd "$CMAKE_DIR" || exit

    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR

    cmake $COMPILE_OPTIONS \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
    -DASCEND_HOME_PATH=$ASCEND_HOME_PATH \
    -DASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR \
    -DCMAKE_PREFIX_PATH="$ASC_CMAKE_DIR" \
    -DASC_DIR="$ASC_CMAKE_DIR" \
    -DSOC_VERSION=Ascend910_9382 \
    -DBUILD_DEEPEP_MODULE=$BUILD_DEEPEP_MODULE \
    -DBUILD_KERNELS_MODULE=$BUILD_KERNELS_MODULE \
    -B "$BUILD_DIR" \
    -S .

    cmake --build "$BUILD_DIR" --target install -j 16
    cd -
}

function build_deepep_kernels()
{
    if [[ "$ONLY_BUILD_DEEPEP_ADAPTER_MODULE" == "ON" ]]; then return 0; fi
    if [[ "$BUILD_DEEPEP_MODULE" != "ON" ]]; then return 0; fi

    if [[ "$BUILD_DEEPEP_OPS" == "ON" ]]; then
        KERNEL_DIR="csrc/deepep/ops"
    else
        KERNEL_DIR="csrc/deepep/ops2"
    fi
    CUSTOM_OPP_DIR="${CURRENT_DIR}/python/deep_ep/deep_ep"

    cd "$KERNEL_DIR" || exit

    chmod +x build.sh
    ./build.sh

    custom_opp_file=$(find ./build_out -maxdepth 1 -type f -name "custom_opp*.run")
    if [ -z "$custom_opp_file" ]; then
        echo "can not find run package"
        exit 1
    else
        echo "find run package: $custom_opp_file"
        chmod +x "$custom_opp_file"
    fi
    rm -rf "$CUSTOM_OPP_DIR"/vendors
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
    mv -v "$CURRENT_DIR"/contrib/torch_memory_saver/python/dist/torch_memory_saver*.whl "${OUTPUT_DIR}/"
    rm -rf "$CURRENT_DIR"/contrib/torch_memory_saver/python/dist
    cd -
}

function create_deepep_cmake()
{
    if [[ "$BUILD_DEEPEP_MODULE" != "ON" ]]; then return 0; fi

    cd csrc || exit
    chmod +x deepep_cmake_build.sh
    chmod +x deepep/build.sh
    chmod +x deepep/compile_ascend_proj.sh
    echo "${FUNCNAME[0]}:./deepep_cmake_build.sh all $SOC_VERSION"
    ./deepep_cmake_build.sh all $SOC_VERSION

    if [[ "$BUILD_DEEPEP_OPS" == "ON" ]]; then
        echo "./deepep/compile_ascend_proj.sh ./deepep $SOC_VERSION deepep"
        bash ./deepep/compile_ascend_proj.sh ./deepep $SOC_VERSION deepep
    else
        echo "./deepep/compile_ascend_proj.sh ./deepep $SOC_VERSION deepep2"
        bash ./deepep/compile_ascend_proj.sh ./deepep $SOC_VERSION deepep2
    fi
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
    cp -v "${CURRENT_DIR}/config.ini" "${CURRENT_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/"
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/python/sgl_kernel_npu/dist/sgl_kernel_npu*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/sgl_kernel_npu/dist
    cd -
}

function build_attentions_kernels()
{
    CUSTOM_OPP_DIR="${CURRENT_DIR}/python/attentions/attentions"
    KERNEL_DIR="csrc/attentions/build"

    cd "$KERNEL_DIR" || exit

    echo "run build attentions library"

    chmod +x build.sh
    ./build.sh
    cd -
}

function make_attentions_package() {
    cd python/attentions || exit

    rm -rf "$CURRENT_DIR"/python/attentions/dist
    python3 setup.py clean --all
    python3 setup.py bdist_wheel
    mv -v "$CURRENT_DIR"/python/attentions/dist/attentions*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/attentions/dist
    cd -
}

function main()
{
    create_deepep_cmake
    build_kernels
    build_deepep_kernels
    build_attentions_kernels
    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel==0.45.1
    fi
    build_memory_saver
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" ]]; then
        make_deepep_package
    fi
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        make_sgl_kernel_npu_package
    fi
    if [[ "$BUILD_ATTENTIONS_MODULE" == "ON" ]]; then
        make_attentions_package
    fi


}

main
