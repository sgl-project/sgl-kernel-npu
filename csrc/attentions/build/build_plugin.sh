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

set -e

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

set +e
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
set -e

USER_ABI_VERSION_RAW=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
if [ $? -ne 0 ]; then
    echo "Error: Failed to retrieve PyTorch ABI version!"
    echo "Possible reasons: PyTorch is not installed, or the version is too old (missing the _GLIBCXX_USE_CXX11_ABI attribute)."
    exit 1
fi
export USER_ABI_VERSION=$(echo "$USER_ABI_VERSION_RAW" | tr -d '[:space:]')


rm -rf build
mkdir -p build
cmake -B build ../csrc
cmake --build build -j

copy_so_files() {
    if [ $# -ne 2 ]; then
        echo "Error: Please pass two arguments (source directory and target directory)"
        return 1
    fi

    local src_dir="$1"
    local dest_dir="$2"

    if [ ! -d "$src_dir" ]; then
        echo "Error: Source directory $src_dir does not exist or is not a valid directory"
        return 1
    fi

    if [ ! -d "$dest_dir" ]; then
        echo "Target directory '$dest_dir' does not exist, creating now..."
        mkdir -p "$dest_dir" || {
            echo "Error: Failed to create target directory $dest_dir"
            return 1
        }
    fi

    echo "Searching for .so files in $src_dir..."
    local so_files=$(find "$src_dir" -type f -name "*.so" 2>/dev/null)

    if [ -z "$so_files" ]; then
        echo "Notice: No .so files found in source directory $src_dir"
        return 0
    fi

    find "$src_dir" -type f -name "*.so" -exec cp {} "$dest_dir" \; 2>/dev/null

    local count=$(echo "$so_files" | wc -l)
    echo "Successfully copied $count .so files to $dest_dir"
    return 0
}

BUILD_DIR=$(dirname $(readlink -f $0))
SRC_DIR=${BUILD_DIR}/build
DSET_DIR=${BUILD_DIR}/../../../python/attentions/attentions/plugin
copy_so_files $SRC_DIR $DSET_DIR
