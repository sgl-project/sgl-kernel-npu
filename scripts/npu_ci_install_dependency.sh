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
WITH_TRITON=false
TRITON_ASCEND_VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cann-version)
            CANN_VERSION="${2:?--cann-version requires a value}"
            shift 2
            ;;
        --with-triton)
            WITH_TRITON=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --cann-version <9.1.0|9.2.0> [--with-triton]"
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
        TRITON_ASCEND_VERSION="3.6.0+dev20261105160244"
        ;;
    "9.1.0")
        TORCH_VERSION="2.10.0"
        TORCHVISION_VERSION="0.25.0"
        TORCHAUDIO_VERSION="2.10.0"
        TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_${ARCHITECT}.whl"
        TRITON_ASCEND_VERSION="3.2.2"
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
    setuptools \
    wheel \
    packaging \
    ninja \
    build \
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


### Install Triton-Ascend
# The artifact must be the same one shipped in the SGLang NPU image for the
# matching CANN version:
#   CANN 9.1.0 -> triton-ascend 3.2.2 (sglang docker/npu.Dockerfile)
#   CANN 9.2.0 -> triton-ascend 3.6.0 dev build (SGLang 26.3 compatibility matrix)
# Override the wheel with TRITON_ASCEND_WHL, e.g. to validate a new nightly.
if [[ "${WITH_TRITON}" == "true" ]]; then
    if [[ -n "${TRITON_ASCEND_WHL:-}" ]]; then
        TRITON_WHEEL="${TRITON_ASCEND_WHL}"
        echo "Installing Triton-Ascend from override: ${TRITON_WHEEL}"
    else
        case "${CANN_VERSION}:${ARCHITECT}" in
            "9.1.0:aarch64")
                # Same artifact as the SGLang NPU image (signed OBS URL, expires 2027-03-29).
                TRITON_WHEEL="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/Triton-ascend/3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1806290330&Signature=eRq3VKjwgP/tTkObtsho%2BzIsmJM%3D"
                ;;
            "9.1.0:x86_64")
                TRITON_WHEEL="https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
                ;;
            "9.2.0:aarch64")
                TRITON_WHEEL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_34_aarch64.whl"
                ;;
            "9.2.0:x86_64")
                TRITON_WHEEL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_34_x86_64.whl"
                ;;
            *)
                echo "Unsupported platform for Triton-Ascend: CANN ${CANN_VERSION}, arch ${ARCHITECT}" >&2
                exit 1
                ;;
        esac
        echo "Installing Triton-Ascend ${TRITON_ASCEND_VERSION} for CANN ${CANN_VERSION} (${ARCHITECT})"
    fi

    ${PIP_INSTALL} "${TRITON_WHEEL}"

    ACTUAL_TRITON_VERSION="$(
        python3 -c 'from importlib.metadata import version; print(version("triton-ascend"))' 2>/dev/null || true
    )"
    if [[ -z "${ACTUAL_TRITON_VERSION}" ]]; then
        echo "Triton-Ascend installation failed: package metadata not found" >&2
        exit 1
    fi
    if [[ -z "${TRITON_ASCEND_WHL:-}" && "${ACTUAL_TRITON_VERSION}" != "${TRITON_ASCEND_VERSION}" ]]; then
        echo "Triton-Ascend version mismatch:" >&2
        echo "  expected: ${TRITON_ASCEND_VERSION}" >&2
        echo "  actual:   ${ACTUAL_TRITON_VERSION}" >&2
        exit 1
    fi

    TRITON_RUNTIME_VERSION="$(python3 -c 'import triton; print(triton.__version__)' 2>/dev/null || echo unknown)"
    echo "Installed Triton-Ascend ${ACTUAL_TRITON_VERSION} (triton runtime ${TRITON_RUNTIME_VERSION})"
fi
