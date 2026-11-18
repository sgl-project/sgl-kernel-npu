#!/bin/bash
set -e

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
fi

cd "${GITHUB_WORKSPACE}"
export UV_SYSTEM_PYTHON=true
export UV_NO_CACHE=1

# --- Step 1: Install all dependencies BEFORE build (per developer recommendation) ---

# Triton-Ascend (CANN-customized triton with triton.language.extra.cann) is
# installed by npu_ci_install_dependency.sh --with-triton so that the wheel
# matches the SGLang NPU image for the same CANN version.
if ! python3 -c 'from importlib.metadata import version; version("triton-ascend")' >/dev/null 2>&1; then
    echo "ERROR: Triton-Ascend is not installed. Run scripts/npu_ci_install_dependency.sh --cann-version <9.1.0|9.2.0> --with-triton first." >&2
    exit 1
fi

# Install other test dependencies
uv pip install expecttest einops pytest packaging ninja

# --- Step 2: Build and install kernel module ---

case "${NPU_HARDWARE:-}" in
    a3) ARCH_VERSION=2201 ;;
    950) ARCH_VERSION=3510 ;;
    *)
        echo "Unsupported NPU_HARDWARE: ${NPU_HARDWARE:-unset} (expected a3 or 950)" >&2
        exit 1
        ;;
esac

echo '=== npu-smi info -t board -i 0 ==='
probe_status=0
npu-smi info -t board -i 0 2>&1 || probe_status=$?
echo "npu-smi exit code: $probe_status"

echo '=== npu-smi info -t board -i 0 -c 0 ==='
probe_status=0
npu-smi info -t board -i 0 -c 0 2>&1 || probe_status=$?
echo "npu-smi exit code: $probe_status"

python3 setup.py --arch-version "$ARCH_VERSION" --artifact wheel
python3 -m pip install output/sgl_kernel_npu-*.whl --no-cache-dir --force-reinstall --no-deps

# --- Step 3: CI workarounds for test-side issues ---

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

# 2. triton.language.parallel compatibility: decode_attention.py uses
#    tl.parallel(0, 2, bind_sub_block=True). The triton frontend pulled in by
#    Triton-Ascend (3.5.0 for 3.2.2 on aarch64, 3.6.0 for 9.2.0) does not
#    export it, so patch triton/language/__init__.py with a range shim.
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
TRITON_LANG_INIT="$PYTHON_SITE/triton/language/__init__.py"
if [ -f "$TRITON_LANG_INIT" ] && ! grep -q "def parallel" "$TRITON_LANG_INIT" 2>/dev/null; then
    cat >> "$TRITON_LANG_INIT" << 'PYEOF'

# CI compatibility shim: tl.parallel is absent from the triton frontend used by Triton-Ascend
# decode_attention.py uses tl.parallel(0, 2, bind_sub_block=True) for sub-block iteration
def parallel(*args, **kwargs):
    import builtins
    return builtins.range(*args)
PYEOF
    echo "Added tl.parallel compatibility shim to triton/language/__init__.py"
fi
