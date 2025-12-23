#!/usr/bin/env bash
set -euo pipefail

export ARCHITECT="$(arch)"
export DEBIAN_FRONTEND="noninteractive"
export PIP_INSTALL="python3 -m pip install --no-cache-dir"


### Dependency Verisons
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCH_NPU_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/torch_npu/torch_npu-2.8.0.post2.dev20251113-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"


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
    curl \
    zip \
    unzip

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


### Install pytorch
## torch
${PIP_INSTALL} \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cpu
## torch_npu
${PIP_INSTALL} ${TORCH_NPU_URL}
