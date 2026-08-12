#!/bin/bash
set -e

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
fi

cd "${GITHUB_WORKSPACE}"
export BUILD_CATLASS_MODULE=ON
bash build.sh -a kernels
pip install ${GITHUB_WORKSPACE}/output/sgl_kernel_npu*.whl --no-cache-dir --force-reinstall --no-deps

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

# --- CI workarounds for test-side issues ---

# 1. sglang: test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py imports
#    'from sglang.srt.utils import is_npu'. The sglang docker image has sglang
#    pre-installed. Try to use it; if import fails (e.g., torch version mismatch
#    after npu_ci_install_dependency.sh), fall back to a minimal stub.
if python3 -c "from sglang.srt.utils import is_npu" 2>/dev/null; then
    echo "sglang is available from docker image"
else
    pip uninstall sglang -y 2>/dev/null || true
    PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    rm -rf "$PYTHON_SITE/sglang" 2>/dev/null || true
    mkdir -p "$PYTHON_SITE/sglang/srt"
    touch "$PYTHON_SITE/sglang/__init__.py" "$PYTHON_SITE/sglang/srt/__init__.py"
    printf 'def is_npu():\n    try:\n        import torch_npu\n        return True\n    except ImportError:\n        return False\n' \
        > "$PYTHON_SITE/sglang/srt/utils.py"
    echo "sglang import failed, created stub for is_npu"
fi

# 2. F (torch.nn.functional): test_swiglu_quant.py uses F.silu() without
#    'import torch.nn.functional as F'. Inject F as a builtin via
#    sitecustomize.py so direct 'python3 test_file.py' invocations work.
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
SITECUSTOMIZE="$PYTHON_SITE/sitecustomize.py"
if ! grep -q "builtins.F" "$SITECUSTOMIZE" 2>/dev/null; then
    cat >> "$SITECUSTOMIZE" << 'PYEOF'
# CI workaround: inject torch.nn.functional as F builtin
try:
    import builtins
    import torch.nn.functional as F
    builtins.F = F
except Exception:
    pass
PYEOF
    echo "Added F injection to sitecustomize.py"
fi
