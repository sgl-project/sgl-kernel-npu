#!/usr/bin/env bash
set -euo pipefail

export ARCHITECT="$(arch)"
export DEBIAN_FRONTEND="noninteractive"
export PIP_INSTALL="python3 -m pip install --no-cache-dir"
export UV_PIP_INSTALL="uv pip install"


### Dependency Versions
# PyTorch: Default to torch 2.8.0, can be overridden by --torch-version

# Default values
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION=""
CANN_VERSION=""
TORCH_NPU_URL=""

USE_CANN_VERSION=false
USE_TORCH_VERSION=false
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cann-version)
            CANN_VERSION="$2"
            USE_CANN_VERSION=true
            shift 2
            ;;
        --torch-version)
            TORCH_VERSION="$2"
            USE_TORCH_VERSION=true
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage:"
            echo "  $0 --cann-version <9.0.0|9.1.0>"
            echo "  $0 --torch-version <2.8.0|2.10.0>"
            echo "Usage: $0 [--cann-version <9.0.0|9.1.0>]"
            exit 1
            ;;
    esac
done

if [[ "${USE_CANN_VERSION}" == true && "${USE_TORCH_VERSION}" == true ]]; then
    echo "Error: --cann-version and --torch-version cannot be used together"
    exit 1
fi


# New pipeline logic
if [[ "${USE_CANN_VERSION}" == true ]]; then

    case "${CANN_VERSION}" in
        "9.0.0")
            TORCH_VERSION="2.10.0"
            TORCHVISION_VERSION="0.25.0"
            TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"
            ;;
        "9.1.0")
            TORCH_VERSION="2.10.0"
            TORCHVISION_VERSION="0.25.0"
            TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_${ARCHITECT}.whl"
            ;;
        *)
            echo "Unsupported CANN version: ${CANN_VERSION}"
            echo "Supported versions: 9.0.0, 9.1.0"
            exit 1
            ;;
    esac


# Old pipeline logic
elif [[ "${USE_TORCH_VERSION}" == true ]]; then

    case "${TORCH_VERSION}" in
        "2.8.0")
            TORCHVISION_VERSION="0.23.0"
            TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.8.0/torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"
            ;;
        "2.10.0")
            TORCHVISION_VERSION="0.25.0"
            TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"
            ;;
        *)
            echo "Unsupported torch version: ${TORCH_VERSION}"
            echo "Supported versions: 2.8.0, 2.10.0"
            exit 1
            ;;
    esac


# No parameter, keep old default behavior
else

    case "${TORCH_VERSION}" in
        "2.8.0")
            TORCHVISION_VERSION="0.23.0"
            TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.8.0/torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"
            ;;
        *)
            echo "Unsupported default torch version: ${TORCH_VERSION}"
            exit 1
            ;;
    esac

fi
case "${CANN_VERSION}" in
    "9.0.0")
        TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp311-cp311-manylinux_2_28_${ARCHITECT}.whl"
        ;;
    "9.1.0")
        TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_${ARCHITECT}.whl"
        ;;
    *)
        echo "Unsupported CANN version: ${CANN_VERSION}"
        echo "Supported versions: 9.0.0, 9.1.0"
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
${PIP_INSTALL} uv
export UV_NO_CACHE=true
export UV_SYSTEM_PYTHON=true
export UV_INDEX_STRATEGY=unsafe-best-match
${UV_PIP_INSTALL} \
    pybind11 \
    pyyaml \
    decorator \
    scipy \
    attrs \
    psutil


### Install pytorch
## torch
${UV_PIP_INSTALL} \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} \
    --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
## torch_npu
# GitCode does not allow UV downloads.
${PIP_INSTALL} ${TORCH_NPU_URL}
