#!/bin/bash
set -e

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly OUTPUT_DIR="${PROJECT_ROOT}/output"
readonly BUILD_DIR="${PROJECT_ROOT}/build"
readonly DEFAULT_ASCEND_ROOT="/usr/local/Ascend/ascend-toolkit"

export VERSION="${VERSION:-1.0.0}"

BUILD_TARGET="all"
REQUESTED_SOC_VERSION=""
SOC_VERSION=""
CMAKE_SOC_VERSION="Ascend910_9382"
SGL_KERNEL_NPU_BUILD_TARGET="Ascend910"

BUILD_ATTENTIONS_MODULE="OFF"
BUILD_KERNELS_MODULE="OFF"
BUILD_MEMORY_SAVER_MODULE="OFF"
BUILD_CATLASS_MODULE="${BUILD_CATLASS_MODULE:-OFF}"

DEBUG_MODE="OFF"
ASC_CMAKE_DIR=""
ASCEND_INCLUDE_DIR=""

function die()
{
    echo "Error: $*" 1>&2
    exit 1
}

function print_help()
{
    cat <<'EOF'
Usage:
    ./build.sh [SOC_VERSION]                    Build all modules; auto-detect A2 or A3.
    ./build.sh -a kernels [SOC_VERSION]         Build sgl_kernel_npu; auto-detect A2, A3, or Ascend 950.
    ./build.sh -a memory-saver                  Build torch_memory_saver.
    ./build.sh -a attentions                    Build attentions.

Omitting SOC_VERSION auto-detects it with npu-smi; hosts without a device fall
back to Ascend910_9382, the value every target used to default to.

TARGET:
    all            (default) Build all modules.
    kernels        Build sgl_kernel_npu only.
    memory-saver   Build torch_memory_saver only.
    attentions     Build attentions only.

SOC_VERSION:
    Ascend910B1         A2 chip. Valid for all/kernels.
    Ascend910_9382      A3 chip. Valid for all/kernels.
    Ascend950           Ascend 950 family alias. Valid for kernels.
    Ascend950PR_9599    Concrete Ascend 950 compiler target. Valid for kernels.
    (omitted)           all/kernels: auto-detect via npu-smi,
                        fallback to Ascend910_9382.

SOC_VERSION aliases:
    910B | Ascend910B1                       A2, native Gemma provider
    910  | 910C | Ascend910_9382             A3, native Gemma provider
    950  | Ascend950 | Ascend950{PR,DT}_*    Ascend 950, ACLNN Gemma provider

For kernels, generic and detected Ascend 950 names compile with Ascend950PR_9599,
the target of the 950 release package. Pass a concrete Ascend950PR_* or
Ascend950DT_* target to override it; every Ascend 950 form selects the Ascend950
wheel provider. The full build does not support Ascend 950: attentions only covers
910-series compute units, so build the modules separately.

Options:
    -d             Enable debug logging.
    -h             Show this help.

Examples:
    ./build.sh                                  # all modules, A3
    ./build.sh -a kernels                       # sgl_kernel_npu, A3
    ./build.sh -a kernels Ascend950PR_9599      # sgl_kernel_npu, A5
    ./build.sh -a memory-saver                  # torch_memory_saver
EOF
}

function parse_arguments()
{
    local opt
    OPTIND=1

    while getopts ":a:hd" opt; do
        case "$opt" in
            a )
                BUILD_TARGET="$OPTARG"
                ;;
            d )
                DEBUG_MODE="ON"
                ;;
            h )
                print_help
                exit 0
                ;;
            \? )
                die "Unknown flag: -$OPTARG. Run './build.sh -h' for more information."
                ;;
            : )
                die "-$OPTARG requires a value. Run './build.sh -h' for more information."
                ;;
        esac
    done

    shift $((OPTIND - 1))
    if [[ "$#" -gt 1 ]]; then
        die "Expected at most one SOC_VERSION argument."
    fi
    REQUESTED_SOC_VERSION="${1:-}"
}

function configure_build_target()
{
    case "$BUILD_TARGET" in
        all )
            BUILD_ATTENTIONS_MODULE="ON"
            BUILD_KERNELS_MODULE="ON"
            BUILD_MEMORY_SAVER_MODULE="ON"
            ;;
        kernels )
            BUILD_KERNELS_MODULE="ON"
            ;;
        memory-saver )
            BUILD_MEMORY_SAVER_MODULE="ON"
            ;;
        attentions )
            BUILD_ATTENTIONS_MODULE="ON"
            ;;
        * )
            die "Invalid target '$BUILD_TARGET'. Allowed values: all|kernels|memory-saver|attentions"
            ;;
    esac
}

# A chip this script does not recognize falls back to the A3 target that used to
# be the unconditional default, so a build that never failed before still cannot
# fail here.
function detect_soc_version()
{
    local board_info=""

    # Build containers may not expose an NPU. Preserve the existing A3 default
    # in that case; callers can still provide Ascend950 explicitly.
    if ! command -v npu-smi >/dev/null 2>&1; then
        SOC_VERSION="Ascend910_9382"
        echo "Cannot find npu-smi; defaulting SOC_VERSION to $SOC_VERSION"
        return
    fi

    # A5 supports the device-level query and reports Chip Name as Ascend950*.
    board_info="$(npu-smi info -t board -i 0 2>/dev/null || true)"
    if printf '%s\n' "$board_info" |
        grep -Eiq '^[[:space:]]*Chip Name[[:space:]]*:[[:space:]]*Ascend950'; then
        SOC_VERSION="Ascend950"
        echo "Detected A5: SOC_VERSION=$SOC_VERSION"
        return
    fi

    # A2 and A3 require a chip ID. Check A2 first because it reports 910B*,
    # while A3 reports Ascend910.
    board_info="$(npu-smi info -t board -i 0 -c 0 2>/dev/null || true)"
    if printf '%s\n' "$board_info" |
        grep -Eiq '^[[:space:]]*Chip Name[[:space:]]*:[[:space:]]*910B'; then
        SOC_VERSION="Ascend910B1"
        echo "Detected A2: SOC_VERSION=$SOC_VERSION"
        return
    fi

    if printf '%s\n' "$board_info" |
        grep -Eiq '^[[:space:]]*Chip Name[[:space:]]*:[[:space:]]*Ascend910'; then
        SOC_VERSION="Ascend910_9382"
        echo "Detected A3: SOC_VERSION=$SOC_VERSION"
        return
    fi

    SOC_VERSION="Ascend910_9382"
    echo "Cannot recognize the NPU from npu-smi output; falling back to SOC_VERSION=$SOC_VERSION"
}

# Fold friendly product names onto canonical SoC names. Kernel-only builds keep
# an explicit concrete Ascend 950 compiler target; the full build keeps the
# generic Ascend950 family name, which it rejects below.
function normalize_soc_version()
{
    case "$SOC_VERSION" in
        910 | Ascend910 | 910C )
            SOC_VERSION="Ascend910_9382"
            ;;
        910B )
            SOC_VERSION="Ascend910B1"
            ;;
        950 | [Aa]scend950 )
            SOC_VERSION="Ascend950"
            ;;
        [Aa]scend950[Pp][Rr]_* )
            if [[ "$BUILD_TARGET" == "kernels" ]]; then
                SOC_VERSION="Ascend950PR_${SOC_VERSION#*_}"
            else
                SOC_VERSION="Ascend950"
            fi
            ;;
        [Aa]scend950[Dd][Tt]_* )
            if [[ "$BUILD_TARGET" == "kernels" ]]; then
                SOC_VERSION="Ascend950DT_${SOC_VERSION#*_}"
            else
                SOC_VERSION="Ascend950"
            fi
            ;;
        * )
            # Anything else is a concrete AscendC compiler target (Ascend910B2,
            # Ascend910_9391, ...). Pass it through to CMake untouched; the
            # AscendC toolchain rejects a target it cannot compile for.
            ;;
    esac
}

function configure_soc_version()
{
    case "$BUILD_TARGET" in
        all | kernels )
            # No SOC_VERSION given: ask the local NPU. detect_soc_version falls
            # back to the A3 target when npu-smi is absent, so build containers
            # without a device keep producing the same artifacts as before.
            if [[ -n "$REQUESTED_SOC_VERSION" ]]; then
                SOC_VERSION="$REQUESTED_SOC_VERSION"
            else
                detect_soc_version
            fi
            ;;
        memory-saver )
            if [[ -n "$REQUESTED_SOC_VERSION" ]]; then
                die "Target 'memory-saver' does not accept SOC_VERSION."
            fi
            SOC_VERSION=""
            ;;
    esac

    if [[ -n "$SOC_VERSION" ]]; then
        normalize_soc_version
    fi

    # 'kernels' has no allow-list: whatever the caller passed goes to the AscendC
    # compiler. AscendC needs a concrete SoC, and csrc/CMakeLists.txt derives
    # arch35 and the Ascend 950-only ops from it, so the generic Ascend950 family
    # name becomes the release package's Ascend950PR_9599 (build_and_release.yml);
    # explicit Ascend950PR_* and Ascend950DT_* values pass through unchanged.
    #
    # 'all' also builds attentions, whose AscendC ops cover only 910-series
    # compute units (csrc/attentions/build/build_ops.sh pins
    # 'ascend910;ascend910b;ascend910_93' and the build entry takes no SoC), so a
    # full Ascend 950 build would silently ship an attentions package without any
    # Ascend 950 op. Refuse the combination instead of mislabelling the output.
    case "$BUILD_TARGET" in
        all )
            if [[ "$SOC_VERSION" == "Ascend950" ]]; then
                die "The full build does not support Ascend 950." \
                    "attentions only covers 910-series compute units; run './build.sh -a kernels 950', './build.sh -a attentions' and './build.sh -a memory-saver' separately."
            fi
            CMAKE_SOC_VERSION="$SOC_VERSION"
            ;;
        kernels )
            CMAKE_SOC_VERSION="$SOC_VERSION"
            if [[ "$SOC_VERSION" == "Ascend950" ]]; then
                CMAKE_SOC_VERSION="Ascend950PR_9599"
            fi
            ;;
    esac

    case "$SOC_VERSION" in
        Ascend950 | Ascend950PR_* | Ascend950DT_* )
            SGL_KERNEL_NPU_BUILD_TARGET="Ascend950"
            ;;
        * )
            SGL_KERNEL_NPU_BUILD_TARGET="Ascend910"
            ;;
    esac
    export SGL_KERNEL_NPU_BUILD_TARGET

    echo "Build target: $BUILD_TARGET"
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        echo "CMake SOC_VERSION: $CMAKE_SOC_VERSION"
        echo "Wheel SOC_VERSION: $SGL_KERNEL_NPU_BUILD_TARGET"
    fi
}

function resolve_ascend_home()
{
    local configured_path="${ASCEND_HOME_PATH:-}"
    local candidate=""
    local path=""

    if [[ -f "$configured_path" && "$(basename "$configured_path")" == "set_env.sh" ]]; then
        configured_path="$(dirname "$configured_path")"
    fi

    if [[ -n "$configured_path" && "$(basename "$configured_path")" == "latest" ]]; then
        candidate="$(readlink -f "$configured_path" 2>/dev/null || true)"
        if [[ -n "$candidate" ]]; then
            configured_path="$candidate"
        fi
    fi

    if [[ -n "$configured_path" && -d "$configured_path" && -f "$configured_path/set_env.sh" ]]; then
        candidate="$configured_path"
    else
        if [[ -n "$configured_path" ]]; then
            echo "Warning: Ignoring invalid ASCEND_HOME_PATH: $configured_path"
        fi

        candidate=""
        while IFS= read -r path; do
            if [[ -f "$path/set_env.sh" ]]; then
                candidate="$path"
            fi
        done < <(find "$DEFAULT_ASCEND_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            ! -name latest \
            -print 2>/dev/null | sort -V)

        if [[ -z "$candidate" && -d "$DEFAULT_ASCEND_ROOT/latest" ]]; then
            path="$(readlink -f "$DEFAULT_ASCEND_ROOT/latest" 2>/dev/null || true)"
            if [[ -n "$path" && -f "$path/set_env.sh" ]]; then
                candidate="$path"
            fi
        fi

        if [[ -z "$candidate" && -f "$DEFAULT_ASCEND_ROOT/set_env.sh" ]]; then
            candidate="$DEFAULT_ASCEND_ROOT"
        fi
    fi

    if [[ -z "$candidate" ]]; then
        die "Cannot find an Ascend toolkit directory containing set_env.sh"
    fi

    export ASCEND_HOME_PATH="$candidate"
}

function setup_ascend_environment()
{
    local resolved_ascend_home=""
    local asc_config_cmake=""

    resolve_ascend_home
    resolved_ascend_home="$ASCEND_HOME_PATH"

    # CANN's environment script may rewrite ASCEND_HOME_PATH to "latest".
    # Keep the resolved installation directory as the source of truth.
    source "$resolved_ascend_home/set_env.sh"
    export ASCEND_HOME_PATH="$resolved_ascend_home"
    export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$resolved_ascend_home}"

    asc_config_cmake="$(find "$ASCEND_HOME_PATH" \
        -name "ASCConfig.cmake" \
        -type f \
        -print \
        -quit 2>/dev/null)"
    if [[ -n "$asc_config_cmake" ]]; then
        ASC_CMAKE_DIR="$(dirname "$asc_config_cmake")"
        export CMAKE_PREFIX_PATH="$ASC_CMAKE_DIR${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        export ASC_DIR="$ASC_CMAKE_DIR"
        echo "Found ASCConfig.cmake at: $asc_config_cmake"
    else
        echo "Warning: Cannot find ASCConfig.cmake under $ASCEND_HOME_PATH"
    fi

    ASCEND_INCLUDE_DIR="${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include"

    echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
    echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
    echo "ASCEND_INCLUDE_DIR: $ASCEND_INCLUDE_DIR"
}

function ensure_wheel_package()
{
    if pip3 show wheel >/dev/null 2>&1; then
        echo "wheel has been installed"
    else
        pip3 install wheel==0.45.1
    fi
}

# The top-level CMake project builds the sgl_kernel_npu host library and its
# AscendC kernels, selected by BUILD_KERNELS_MODULE.
function build_cmake_modules()
{
    local cmake_args=(
        "-DCMAKE_INSTALL_PREFIX=$OUTPUT_DIR"
        "-DASCEND_HOME_PATH=$ASCEND_HOME_PATH"
        "-DASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR"
        "-DSOC_VERSION=$CMAKE_SOC_VERSION"
        "-DBUILD_KERNELS_MODULE=$BUILD_KERNELS_MODULE"
        "-DBUILD_CATLASS_MODULE=${BUILD_CATLASS_MODULE:-OFF}"
    )

    if [[ -n "$ASC_CMAKE_DIR" ]]; then
        cmake_args+=(
            "-DCMAKE_PREFIX_PATH=$ASC_CMAKE_DIR"
            "-DASC_DIR=$ASC_CMAKE_DIR"
        )
    fi

    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    cmake "${cmake_args[@]}" \
        -B "$BUILD_DIR" \
        -S "$PROJECT_ROOT"
    cmake --build "$BUILD_DIR" --target install -j 16
}

function build_attentions_kernels()
{
    (
        cd "$PROJECT_ROOT/csrc/attentions/build"
        echo "Building attentions kernels"
        chmod +x build.sh
        ./build.sh
    )
}

function make_sgl_kernel_npu_package()
{
    (
        cd "$PROJECT_ROOT/python/sgl_kernel_npu"
        rm -rf dist
        cp -v "$PROJECT_ROOT/config.ini" sgl_kernel_npu/
        python3 setup.py clean --all
        python3 setup.py bdist_wheel
        mv -v dist/sgl_kernel_npu*.whl "$OUTPUT_DIR/"
        rm -rf dist
    )
}

function make_attentions_package()
{
    (
        cd "$PROJECT_ROOT/python/attentions"
        rm -rf dist
        python3 setup.py clean --all
        python3 setup.py bdist_wheel
        mv -v dist/attentions*.whl "$OUTPUT_DIR/"
        rm -rf dist
    )
}

function build_memory_saver_package()
{
    (
        cd "$PROJECT_ROOT/contrib/torch_memory_saver/python"
        echo "Building torch_memory_saver"
        rm -rf build dist
        python3 setup.py clean --all
        python3 setup.py bdist_wheel
        mv -v dist/torch_memory_saver*.whl "$OUTPUT_DIR/"
        rm -rf dist
    )
}

function main()
{
    parse_arguments "$@"
    configure_build_target
    configure_soc_version
    export DEBUG_MODE

    setup_ascend_environment
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
    echo "CANN path: $ASCEND_HOME_PATH"

    # Build native components first. All module selection is controlled here.
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        build_cmake_modules
    fi
    if [[ "$BUILD_ATTENTIONS_MODULE" == "ON" ]]; then
        build_attentions_kernels
    fi

    ensure_wheel_package

    # Package only the modules selected above.
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        make_sgl_kernel_npu_package
    fi
    if [[ "$BUILD_ATTENTIONS_MODULE" == "ON" ]]; then
        make_attentions_package
    fi
    if [[ "$BUILD_MEMORY_SAVER_MODULE" == "ON" ]]; then
        build_memory_saver_package
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
