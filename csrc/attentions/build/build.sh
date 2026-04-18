#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set -e
BUILD_DIR=$(dirname $(readlink -f $0))
PROJ_ROOT_DIR=${BUILD_DIR}/..
chmod a-w $BUILD_DIR/*

cd ${PROJ_ROOT_DIR}

echo "\nPROJ_ROOT_DIR = ${PROJ_ROOT_DIR}"

PYTHON_VERSION=""
if command -v python3 &> /dev/null; then
    version=$(python3 --version | awk '{print$2}')
    major=$(echo $version | cut -d '.' -f 1)
    minor=$(echo $version | cut -d '.' -f 2)
    PYTHON_VERSION="py${major}${minor}"
    echo "python version is: $PYTHON_VERSION"
else
    echo "cannot get python version"
    exit 1
fi

if [ -n "$PROJ_ROOT_DIR" ] && [ -d "${PROJ_ROOT_DIR}/csrc/ops" ]; then
    source ${PROJ_ROOT_DIR}/build/build_ops.sh ${PROJ_ROOT_DIR}/build
elif [ -n "$PROJ_ROOT_DIR" ]; then
    echo "Waring: The path of custom op operators $PROJ_ROOT_DIR/csrc/ops does not exist."
fi

if [ -n "$PROJ_ROOT_DIR" ] && [ -d "${PROJ_ROOT_DIR}/csrc/plugin" ]; then
    source ${PROJ_ROOT_DIR}/build/build_plugin.sh ${PROJ_ROOT_DIR}/build
elif [ -n "$PROJ_ROOT_DIR" ]; then
    echo "Waring: The path of op plugins $PROJ_ROOT_DIR/csrc/plugin does not exist."
fi

clean_build_dirs() {
    local dirs_to_remove=(
        "${BUILD_DIR}/bdist.linux-aarch64"
        "${BUILD_DIR}/lib"
        "${BUILD_DIR}/output"
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
cd ${PROJ_ROOT_DIR}
