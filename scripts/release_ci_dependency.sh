#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"


# Install the required dependencies in CI.
apt update -y && apt install -y \
    build-essential \
    cmake \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    ca-certificates \
    libtorch-dev

update-ca-certificates
${PIP_INSTALL} --upgrade pip

PYTORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu

PTA_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/torch_npu/torch_npu-2.8.0.post2.dev20251113-cp311-cp311-manylinux_2_28_aarch64.whl"
${PIP_INSTALL} ${PTA_URL}
${PIP_INSTALL} wheel==0.45.1 pybind11
find /usr -name "pybind11Config.cmake"
find / -name types.h 2>/dev/null
ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
source ${ASCEND_CANN_PATH}/set_env.sh
#export LD_LIBRARY_PATH=${ASCEND_CANN_PATH}/latest/aarch64-linux/devlib/libascend_hal.so:$LD_LIBRARY_PATH