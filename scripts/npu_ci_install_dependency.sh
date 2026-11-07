#!/usr/bin/env bash
set -euo pipefail

export ARCHITECT="$(arch)"
export DEBIAN_FRONTEND="noninteractive"
export PIP_INSTALL="python3 -m pip install --no-cache-dir"
export UV_PIP_INSTALL="uv pip install"


### Dependency Versions
# Torch、torchvision 和 torchaudio 根据 CANN 版本自动选择
CANN_VERSION=""
TORCH_VERSION=""
TORCHVISION_VERSION=""
TORCHAUDIO_VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cann-version)
            CANN_VERSION="${2:?--cann-version requires a value}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --cann-version <9.1.0|9.2.0>"
            exit 1
            ;;
    esac
done

case "${CANN_VERSION}" in
    "9.2.0")
        TORCH_VERSION="2.13.0"
        TORCHVISION_VERSION="0.28.0"
        TORCHAUDIO_VERSION="2.11.0"
        TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_${ARCHITECT}.whl"
        ;;
    "9.1.0")
        TORCH_VERSION="2.10.0"
        TORCHVISION_VERSION="0.25.0"
        TORCHAUDIO_VERSION="2.10.0"
        TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_${ARCHITECT}.whl"
        ;;
    *)
        echo "Unsupported CANN version: ${CANN_VERSION}"
        echo "Supported versions: 9.1.0, 9.2.0"
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


### Install PyTorch
## torch
${UV_PIP_INSTALL} \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_CACHE_URL:-https://download.pytorch.org/whl/cpu}" \
    --extra-index-url "${PYPI_CACHE_URL:-https://pypi.org/simple/}"

## torch_npu
# GitCode 不支持通过 uv 下载
${PIP_INSTALL} "${TORCH_NPU_URL}"


### Verify installed torch version
ACTUAL_TORCH_VERSION="$(
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
        python3 -c 'import torch; print(torch.__version__.split("+")[0])'
)"

if [[ "${ACTUAL_TORCH_VERSION}" != "${TORCH_VERSION}" ]]; then
    echo "Torch version mismatch:" >&2
    echo "  expected: ${TORCH_VERSION}" >&2
    echo "  actual:   ${ACTUAL_TORCH_VERSION}" >&2
    exit 1
fi

echo "Installed CANN ${CANN_VERSION} with torch ${ACTUAL_TORCH_VERSION}"
