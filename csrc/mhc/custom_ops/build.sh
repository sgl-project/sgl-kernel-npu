#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

########################################################################################################################
# 预定义变量
########################################################################################################################

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
USER_ID=$(id -u)
PARENT_JOB="false"
HOST_TILING="false"
CHECK_COMPATIBLE="true"
ASAN="false"
UBSAN="false"
COV="false"
CLANG="false"
VERBOSE="false"

PR_CHANGED_FILES=""  # PR场景, 修改文件清单, 可用于标识是否PR场景

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

CUSTOM_OPTION="-DBUILD_OPEN_PROJECT=ON"
CUSTOM_OPTION+=" -DCMAKE_CXX_FLAGS=\"-w\" -DCMAKE_C_FLAGS=\"-w\""

########################################################################################################################
# 预定义函数
########################################################################################################################

function help_info() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo
    echo "-h|--help            Displays help message."
    echo
    echo "-n|--op-name         Specifies the compiled operator. If there are multiple values, separate them with semicolons and use quotation marks. The default is all."
    echo "                     For example: -n \"swiglu_clip_quant\""
    echo
    echo "-c|--compute-unit    Specifies the chip type. If there are multiple values, separate them with semicolons and use quotation marks. The default is ascend910_93."
    echo "                     For example: -c \"ascend910_93\" or -c \"ascend910b\""
    echo
    echo "--tiling_key         Sets the tiling key list for operators. If there are multiple values, separate them with semicolons and use quotation marks. The default is all."
    echo "                     For example: --tiling_key \"1\" or --tiling_key \"1;2;3;4\""
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
    cmake ..  ${CUSTOM_OPTION} ${extra_option}
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

########################################################################################################################
# 参数解析处理
########################################################################################################################

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
    --ccache)
        CCACHE_PROGRAM="$2"
        shift 2
        ;;
    -p|--package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    -b|--build)
        BUILD="$2"
        shift 2
        ;;
    -f|--changed_list)
        PR_CHANGED_FILES="$2"
        shift 2
        ;;
    --parent_job)
        PARENT_JOB="true"
        shift
        ;;
    --enable_host_tiling)
        HOST_TILING="true"
        shift
        ;;
    --disable-check-compatible|--disable-check-compatiable)
        CHECK_COMPATIBLE="false"
        shift
        ;;
    --op_build_tool)
        op_build_tool="$2"
        shift 2
        ;;
    --ascend_cmake_dir)
        ascend_cmake_dir="$2"
        shift 2
        ;;
    --verbose)
        VERBOSE="true"
        shift
        ;;
    --clang)
        CLANG="true"
        shift
        ;;
    --tiling-key|--tiling_key)
        TILING_KEY="$2"
        shift 2
        ;;
    --op_debug_config)
        OP_DEBUG_CONFIG="$2"
        shift 2
        ;;
    --ops-compile-options)
        OPS_COMPILE_OPTIONS="$2"
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

if [ -n "${op_build_tool}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_BUILD_TOOL=${op_build_tool}"
fi

if [ -n "${ascend_cmake_dir}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_CMAKE_DIR=${ascend_cmake_dir}"
fi

if [ -n "${TILING_KEY}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTILING_KEY=${TILING_KEY}"
fi

if [ -n "${OP_DEBUG_CONFIG}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOP_DEBUG_CONFIG=${OP_DEBUG_CONFIG}"
fi

if [ -n "${OPS_COMPILE_OPTIONS}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DOPS_COMPILE_OPTIONS=${OPS_COMPILE_OPTIONS}"
fi

if [ "${HOST_TILING}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_HOST_TILING=true"
fi

if [ -n "${ascend_package_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ascend_package_path}
elif [ -n "${ASCEND_HOME_PATH}" ];then
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

if [ "${PARENT_JOB}" == "false" ]; then
    CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))
    if [ -n "${OPS_CPU_NUMBER}" ]; then
        if [[ "${OPS_CPU_NUMBER}" =~ ^[0-9]+$ ]]; then
            CPU_NUM="${OPS_CPU_NUMBER}"
        fi
    fi
    JOB_NUM="-j${CPU_NUM}"
fi

CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} -DCHECK_COMPATIBLE=${CHECK_COMPATIBLE}"

########################################################################################################################
# 处理流程
########################################################################################################################

set_env

clean

if [ -n "${CCACHE_PROGRAM}" ]; then
    if [ "${CCACHE_PROGRAM}" == "false" ] || [ "${CCACHE_PROGRAM}" == "off" ]; then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=OFF"
    elif [ -f "${CCACHE_PROGRAM}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${CCACHE_PROGRAM}"
        gen_bisheng ${CCACHE_PROGRAM}
    fi
else
    # 判断有无默认的ccache 如果有则使用
    ccache_system=$(which ccache || true)
    if [ -n "${ccache_system}" ];then
        CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_CCACHE=ON -DCUSTOM_CCACHE=${ccache_system}"
        gen_bisheng ${ccache_system}
    fi
fi

cd ${BUILD_DIR}

if [ "${BUILD}" == "host" ];then
    cmake_config -DENABLE_OPS_KERNEL=OFF
    build_host
    # TO DO
    rm -rf ${CURRENT_DIR}/output
    mkdir -p ${CURRENT_DIR}/output
    cp ${BUILD_DIR}/*.run ${CURRENT_DIR}/output
elif [ "${BUILD}" == "kernel" ];then
    cmake_config -DENABLE_OPS_HOST=OFF
    build_kernel
elif [ -n "${BUILD}" ];then
    cmake_config
    build ${BUILD}
else
    cmake_config
    build_package
fi
