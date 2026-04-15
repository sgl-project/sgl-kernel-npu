#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
TARGET_PATH="${CURRENT_DIR}/../mindiesd/ops"
USER_ID=$(id -u)
PARENT_JOB="false"
CHECK_COMPATIBLE="true"
ASAN="false"
COV="false"
VERBOSE="false"

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

CUSTOM_OPTION="-DBUILD_OPEN_PROJECT=ON"

function help_info() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo
    echo "-h|--help            Displays help message."
    echo
    echo "-n|--op-name         Specifies the compiled operator. If there are multiple values, separate them with semicolons and use quotation marks. The default is all."
    echo "                     For example: -n \"flash_attention_score\" or -n \"flash_attention_score;flash_attention_score_grad\""
    echo
    echo "-c|--compute-unit    Specifies the chip type. If there are multiple values, separate them with semicolons and use quotation marks. The default is ascend910b."
    echo "                     For example: -c \"ascend910b\" or -c \"ascend910b;ascend310p\""
    echo
    echo "--cov                Compiles with cov."
    echo
    echo "--verbose            Displays more compilation information."
    echo
}

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

function set_env()
{
    source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash || echo "0"

    export BISHENG_REAL_PATH=$(which bisheng || true)

    if [ -z "${BISHENG_REAL_PATH}" ];then
        log "Error: bisheng compilation tool not found, Please check whether the cann package or environment variables are set."
        exit 1
    fi
}

function clean()
{
    if [ -n "${BUILD_DIR}" ];then
        rm -rf ${BUILD_DIR}
    fi
    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    CMAKE_SOURCE_DIR="${CURRENT_DIR}/../csrc/ops"
    cmake -S ${CMAKE_SOURCE_DIR} -B . ${CUSTOM_OPTION} ${extra_option}
}

function build()
{
    local target="$1"
    if [ "${VERBOSE}" == "true" ];then
        local option="--verbose"
    fi
    cmake --build . --target ${target} ${JOB_NUM} ${option}
}

function gen_bisheng(){
    local ccache_program=$1
    local gen_bisheng_dir=${BUILD_DIR}/gen_bisheng_dir

    if [ ! -d "${gen_bisheng_dir}" ];then
        mkdir -p ${gen_bisheng_dir}
    fi

    pushd ${gen_bisheng_dir}
    $(> bisheng)
    echo "#!/bin/bash" >> bisheng
    echo "ccache_args=""\"""${ccache_program} ${BISHENG_REAL_PATH}""\"" >> bisheng
    echo "args=""$""@" >> bisheng

    if [ "${VERBOSE}" == "true" ];then
        echo "echo ""\"""$""{ccache_args} ""$""args""\"" >> bisheng
    fi

    echo "eval ""\"""$""{ccache_args} ""$""args""\"" >> bisheng
    chmod +x bisheng

    export PATH=${gen_bisheng_dir}:$PATH
    popd
}

function build_package(){
    build package
}

function build_host(){
    build_package
}

function build_kernel(){
    build ops_kernel
}

function build_and_install(){

    log "Info: Looking for .run package in ${OUTPUT_DIR}..."
    RUN_PACKAGE=$(find ${OUTPUT_DIR} -name "CANN-custom_ops-*.run" -type f | head -n 1)

    if [ -z "${RUN_PACKAGE}" ]; then
        log "Error: No .run package found in ${OUTPUT_DIR}!"
        exit 1
    fi
    log "Info: Found .run package: ${RUN_PACKAGE}"

    log "Info: Adding execute permission to ${RUN_PACKAGE}..."
    chmod +x ${RUN_PACKAGE}

    log "Info: install .run package to ${CURRENT_DIR}..."
    # ${RUN_PACKAGE} --extract=${UNPACK_DIR}
    bash ${OUTPUT_DIR}/*.run --install-path=${CURRENT_DIR}

    if [ $? -ne 0 ]; then
        log "Error: Failed to unpack ${RUN_PACKAGE}!"
        exit 1
    fi
    cd ${CURRENT_DIR}


}

while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)
        help_info
        exit
        ;;
    -n|--op-name)
        ascend_op_name="$2"
        shift 2
        ;;
    -c|--compute-unit)
        ascend_compute_unit="$2"
        shift 2
        ;;
    *)
        help_info
        exit 1
        ;;
    esac
done

if [ -n "${ascend_compute_unit}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_COMPUTE_UNIT=${ascend_compute_unit}"
fi

if [ -n "${ascend_op_name}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_OP_NAME=${ascend_op_name}"
fi

if [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    log "Error: Please set the toolkit package installation directory through parameter -p|--package-path."
    exit 1
fi

if [ "${PARENT_JOB}" == "false" ];then
    CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))
    JOB_NUM="-j${CPU_NUM}"
fi

CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} -DCHECK_COMPATIBLE=${CHECK_COMPATIBLE}"

set_env
clean

ccache_system=$(which ccache || true)
if [ -n "${ccache_system}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${ccache_system}"
    gen_bisheng ${ccache_system}
fi




cd ${BUILD_DIR}
cmake_config
build_package
build_and_install

clean_build_dirs() {
    local dirs_to_remove=(
        "${BUILD_DIR}/"
    )

    echo "About to delete the following build-related directories: "
    for dir in "${dirs_to_remove[@]}"; do
        echo "  - $dir"
    done

    for dir in "${dirs_to_remove[@]}"; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
        else
            echo "Directory does not exist, skipping: $dir"
        fi
    done
}

clean_build_dirs



