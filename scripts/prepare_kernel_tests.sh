#!/bin/bash
set -e

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
fi

cd "${GITHUB_WORKSPACE}"
bash build.sh -a kernels
pip install ${GITHUB_WORKSPACE}/output/sgl_kernel_npu*.whl --no-cache-dir
export UV_SYSTEM_PYTHON=true
uv pip install triton expecttest einops pytest
