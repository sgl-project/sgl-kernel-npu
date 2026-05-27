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
# CANN 8.5.0 -> triton-ascend 3.2.0, CANN 9.0.0 -> triton-ascend 3.2.1
if [ -n "${TRITON_ASCEND_WHL:-}" ]; then
    pip install ${TRITON_ASCEND_WHL}
elif [ -n "${CANN_VERSION:-}" ]; then
    case "${CANN_VERSION}" in
        9.0.0)
            pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
            ;;
        *)
            pip install triton-ascend==3.2.0 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
            ;;
    esac
else
    pip install triton-ascend==3.2.0 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
fi

# Install other test dependencies
uv pip install expecttest einops pytest packaging torchair
