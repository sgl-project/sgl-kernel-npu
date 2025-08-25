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


### Install PyTorch and PTA
PYTORCH_VERSION=2.6.0
TORCHVISION_VERSION=0.21.0
${PIP_INSTALL} torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu

PTA_VERSION="v7.1.0.1-pytorch2.6.0"
PTA_NAME="torch_npu-2.6.0.post1-cp311-cp311-manylinux_2_28_aarch64.whl"
PTA_URL="https://gitee.com/ascend/pytorch/releases/download/${PTA_VERSION}/${PTA_NAME}"
wget -O "${PTA_NAME}" "${PTA_URL}" && ${PIP_INSTALL} "./${PTA_NAME}"

### Install Triton-Ascend
TRITON_ASCEND_NAME="triton_ascend-3.2.0.dev20250729-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
TRITON_ASCEND_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/${TRITON_ASCEND_NAME}"
${PIP_INSTALL} attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11
wget -O "${TRITON_ASCEND_NAME}" "${TRITON_ASCEND_URL}" && ${PIP_INSTALL} "./${TRITON_ASCEND_NAME}"
