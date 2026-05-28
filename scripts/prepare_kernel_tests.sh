#!/bin/bash
set -e

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
fi

cd "${GITHUB_WORKSPACE}"
bash build.sh -a kernels
pip install ${GITHUB_WORKSPACE}/output/sgl_kernel_npu*.whl --no-cache-dir

export UV_SYSTEM_PYTHON=true

# Install Triton-Ascend (CANN-customized triton with triton.language.extra.cann)
# Official version mapping (strict 1:1):
#   CANN 8.5.0 -> triton-ascend 3.2.0
#   CANN 9.0.0 -> triton-ascend 3.2.1
if [ -n "${TRITON_ASCEND_WHL:-}" ]; then
    pip install ${TRITON_ASCEND_WHL}
else
    CANN_VER="${CANN_VERSION:-8.5.0}"
    case "$CANN_VER" in
        8.5.*)
            TRITON_ASCEND_VER="3.2.0"
            ;;
        9.0.*)
            TRITON_ASCEND_VER="3.2.1"
            ;;
        *)
            echo "WARNING: Unknown CANN version $CANN_VER, defaulting to triton-ascend 3.2.0"
            TRITON_ASCEND_VER="3.2.0"
            ;;
    esac
    echo "Installing triton-ascend==${TRITON_ASCEND_VER} for CANN ${CANN_VER}"
    pip install triton-ascend==${TRITON_ASCEND_VER} --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
fi

# Install other test dependencies
uv pip install expecttest einops pytest packaging