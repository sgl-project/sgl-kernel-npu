#!/bin/bash
set -e

if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    git config --global --add safe.directory "${GITHUB_WORKSPACE}"
fi

cd "${GITHUB_WORKSPACE}"

WHEEL_PATH=""
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || "$1" != --wheel ]]; then
        echo "Usage: $0 [--wheel PATH]" >&2
        exit 2
    fi
    WHEEL_PATH="$2"
    if [[ ! -f "$WHEEL_PATH" ]]; then
        echo "Kernel wheel does not exist: $WHEEL_PATH" >&2
        exit 1
    fi
fi

export UV_SYSTEM_PYTHON=true
export UV_NO_CACHE=1

# --- Step 1: Install test and build dependencies ---

# Install Triton-Ascend (CANN-customized triton with triton.language.extra.cann).
# Keep the wheel identical to the SGLang NPU image for the same CANN version:
#   CANN 9.1.0 -> triton-ascend 3.2.2
#   CANN 9.2.0 -> triton-ascend 3.6.0 dev build
# CANN 8.5.0 and 9.0.0 are no longer supported.
if [ -n "${TRITON_ASCEND_WHL:-}" ]; then
    TRITON_ASCEND_VER="custom"
    echo "Installing Triton-Ascend from override: ${TRITON_ASCEND_WHL}"
    pip install "${TRITON_ASCEND_WHL}"
else
    CANN_VER="${CANN_VERSION:-}"
    ARCH="$(arch)"
    case "$CANN_VER" in
        9.1.*)
            TRITON_ASCEND_VER="3.2.2"
            ;;
        9.2.*)
            TRITON_ASCEND_VER="3.6.0+dev20261105160244"
            ;;
        *)
            echo "Unsupported CANN version '${CANN_VER}' for Triton-Ascend; supported versions: 9.1.x, 9.2.x" >&2
            exit 1
            ;;
    esac

    case "${TRITON_ASCEND_VER}:${ARCH}" in
        "3.2.2:aarch64")
            # Same artifact as the SGLang NPU image (signed OBS URL, expires 2027-03-29).
            TRITON_ASCEND_URL="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/Triton-ascend/3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1806290330&Signature=eRq3VKjwgP/tTkObtsho%2BzIsmJM%3D"
            ;;
        "3.2.2:x86_64")
            TRITON_ASCEND_URL="https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
            ;;
        "3.6.0+dev20261105160244:aarch64")
            TRITON_ASCEND_URL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_34_aarch64.whl"
            ;;
        "3.6.0+dev20261105160244:x86_64")
            TRITON_ASCEND_URL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_34_x86_64.whl"
            ;;
        *)
            echo "Unsupported architecture for Triton-Ascend: ${ARCH}" >&2
            exit 1
            ;;
    esac

    echo "Installing Triton-Ascend ${TRITON_ASCEND_VER} for CANN ${CANN_VER} (${ARCH})"
    pip install "${TRITON_ASCEND_URL}"
fi

ACTUAL_TRITON_VERSION="$(
    python3 -c 'from importlib.metadata import version; print(version("triton-ascend"))' 2>/dev/null || true
)"
if [ -z "$ACTUAL_TRITON_VERSION" ]; then
    echo "Triton-Ascend installation failed: package metadata not found" >&2
    exit 1
fi
if [ "$TRITON_ASCEND_VER" != "custom" ] && [ "$ACTUAL_TRITON_VERSION" != "$TRITON_ASCEND_VER" ]; then
    echo "Triton-Ascend version mismatch: expected ${TRITON_ASCEND_VER}, got ${ACTUAL_TRITON_VERSION}" >&2
    exit 1
fi
TRITON_RUNTIME_VERSION="$(python3 -c 'import triton; print(triton.__version__)' 2>/dev/null || echo unknown)"
echo "Installed Triton-Ascend ${ACTUAL_TRITON_VERSION} (triton runtime ${TRITON_RUNTIME_VERSION})"

# Install other test dependencies
uv pip install expecttest einops pytest packaging ninja

# --- Step 2: Build a wheel or install an existing CI artifact ---

if [[ -z "$WHEEL_PATH" ]]; then
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
    kernel_wheels=(output/sgl_kernel_npu-*.whl)
else
    echo "Using prebuilt kernel wheel: $WHEEL_PATH"
    kernel_wheels=("$WHEEL_PATH")
fi
python3 -m pip install "${kernel_wheels[@]}" --no-cache-dir --force-reinstall --no-deps

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
