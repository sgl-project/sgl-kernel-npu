#!/bin/bash
set -euo pipefail

PIP_INSTALL="pip install --no-cache-dir"


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
    ca-certificates
update-ca-certificates
python3 -m ${PIP_INSTALL} --upgrade pip

PYTORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu

PTA_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/torch_npu/torch_npu-2.8.0.post2.dev20251113-cp311-cp311-manylinux_2_28_aarch64.whl"
${PIP_INSTALL} ${PTA_URL}
${PIP_INSTALL} wheel==0.45.1 pybind11
ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
source ${ASCEND_CANN_PATH}/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=~/.local/lib/python3.11/site-packages/pybind11/share/camke/pybind11
bash build.sh