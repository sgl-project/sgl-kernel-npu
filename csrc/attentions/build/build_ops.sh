#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# 构建环境使用CANN主线包，容易引入兼容性问题。同时为了更好地控制对外发布内容，我们
# 在构建环境用msopgen工具生成工程，然后将要发布的算子交付件拷贝到新生成的工程构建
set -e

current_script_dir=$(realpath $1)

if [ -n "${ASCEND_TOOLKIT_HOME}" ]; then
    local_toolkit=${ASCEND_TOOLKIT_HOME}
    echo "Using ASCEND_TOOLKIT_HOME: ${local_toolkit}"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    local_toolkit=/usr/local/Ascend/ascend-toolkit/latest
    echo "Using default toolkit path: ${local_toolkit}"
elif [ -d "/home/slave1/Ascend/ascend-toolkit/latest" ]; then
    local_toolkit=/home/slave1/Ascend/ascend-toolkit/latest
    echo "Using alternative toolkit path: ${local_toolkit}"
else
    echo "Can not find toolkit path, please set ASCEND_TOOLKIT_HOME"
    echo "eg: export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest"
    exit 1
fi

msopgen=${local_toolkit}/python/site-packages/bin/msopgen
if [ ! -f ${msopgen} ]; then
    echo "${msopgen} not exists"
    exit 1
fi

function build_ops(){
    ori_path=${PWD}
    chmod -R u+w ${current_script_dir}
    cd ${current_script_dir}
    rm -rf vendors
    source ${current_script_dir}/build_ascendc_ops.sh -n 'laser_attention;block_sparse_attention;sparse_block_estimate' -c 'ascend910;ascend910b;ascend910_93'
    rm -rf ${current_script_dir}/vendors/aie_ascendc/bin
    rm -rf ${current_script_dir}/vendors/customize/bin
    cd ${current_script_dir}
}

copy_ops() {
    SRC_DIR="${current_script_dir}/vendors"
    DST_DIR="${current_script_dir}/../../../python/attentions/attentions/ops/vendors"

    echo "Source directory: $SRC_DIR"
    echo "Destination directory: $DST_DIR"

    # Check source directory
    if [ ! -d "$SRC_DIR" ]; then
        echo "Error: source directory $SRC_DIR does not exist!"
        return 1
    fi

    # Create destination directory
    mkdir -p "$DST_DIR"

    # (Optional) Clean the target directory
    echo "Cleaning destination directory..."
    rm -rf "${DST_DIR:?}/"*

    echo "Copying all subdirectories under pkg to attentions/ops..."
    for subdir in "$SRC_DIR"/*; do
        if [ -d "$subdir" ]; then
            echo "Copying directory: $subdir → $DST_DIR"
            cp -a "$subdir" "$DST_DIR/"
        fi
    done
    echo "Copy finished!"
}

build_ops
copy_ops
