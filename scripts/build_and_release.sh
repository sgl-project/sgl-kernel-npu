#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PIP_INSTALL="python3 -m pip install --no-cache-dir"


### Install required dependencies
## APT packages
apt update -y && \
apt upgrade -y && \
apt install -y \
    locales \
    ca-certificates \
    build-essential \
    cmake \
    ccache \
    pkg-config \
    zlib1g-dev \
    wget \
    curl

## Setup
locale-gen en_US.UTF-8
update-ca-certificates
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8

## Python packages
${PIP_INSTALL} --upgrade pip
# Pin wheel to 0.45.1, REF: https://github.com/pypa/wheel/issues/662
${PIP_INSTALL} \
    wheel==0.45.1 \
    pybind11


### Install PyTorch
## PyTorch
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
${PIP_INSTALL} \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cpu

## PTA
PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_aarch64.whl"
${PIP_INSTALL} ${PTA_URL}


### Build SGL-Kernel-NPU
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH}
bash build.sh
