#!/usr/bin/env bash
set -euo pipefail

export ARCHITECT="$(arch)"
export DEBIAN_FRONTEND="noninteractive"
export PIP_INSTALL="python3 -m pip install --no-cache-dir"


### Dependency Versions
# PyTorch: Default to torch 2.8.0, can be overridden by --torch-version
TORCH_VERSION="2.8.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --torch-version)
            TORCH_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--torch-version <2.8.0|2.10.0>]"
            exit 1
            ;;
    esac
done

case "${TORCH_VERSION}" in
    "2.8.0")
        TORCHVISION_VERSION="0.23.0"
        TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"
        ;;
    "2.10.0")
        TORCHVISION_VERSION="0.25.0"
        TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/7.3.0.alpha002/torch_npu-2.10.0rc2-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"
        ;;
    *)
        echo "Unsupported torch version: ${TORCH_VERSION}"
        echo "Supported versions: 2.8.0, 2.10.0"
        exit 1
        ;;
esac


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
    pybind11 \
    pyyaml \
    decorator \
    scipy \
    attrs \
    psutil


### Install pytorch
## torch
${PIP_INSTALL} \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} \
    --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
## torch_npu
${PIP_INSTALL} ${TORCH_NPU_URL}
