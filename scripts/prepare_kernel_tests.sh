#!/bin/bash
set -e

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
fi

cd "${GITHUB_WORKSPACE}"
export BUILD_CATLASS_MODULE=ON
bash build.sh -a kernels
pip install ${GITHUB_WORKSPACE}/output/sgl_kernel_npu*.whl --no-cache-dir

export UV_SYSTEM_PYTHON=true

# Install Triton-Ascend (provides triton.language.extra.cann).
# Per developer guidance, use 3.2.1 for ALL CANN versions (including 8.5.0)
# because 3.2.0 lacks the cann extra module, causing ImportError in many tests.
if [ -n "${TRITON_ASCEND_WHL:-}" ]; then
    pip install ${TRITON_ASCEND_WHL}
else
    echo "Installing triton-ascend==3.2.1 for CANN ${CANN_VERSION:-unknown}"
    pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
fi

# Install other test dependencies
uv pip install expecttest einops pytest packaging

# --- CI workarounds for test-side issues ---

# 1. sglang: test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py imports
#    'from sglang.srt.utils import is_npu'. Full sglang pulls torch-memory-saver
#    (needs CUDA, unavailable in NPU containers) and its __init__.py imports
#    orjson etc. that are also missing with --no-deps. Create a clean minimal
#    stub that ONLY provides is_npu, removing any partial sglang install first.
pip uninstall sglang -y 2>/dev/null || true
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
rm -rf "$PYTHON_SITE/sglang" 2>/dev/null || true
mkdir -p "$PYTHON_SITE/sglang/srt"
touch "$PYTHON_SITE/sglang/__init__.py" "$PYTHON_SITE/sglang/srt/__init__.py"
printf 'def is_npu():\n    try:\n        import torch_npu\n        return True\n    except ImportError:\n        return False\n' \
    > "$PYTHON_SITE/sglang/srt/utils.py"
echo "Created clean sglang stub for is_npu"

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
