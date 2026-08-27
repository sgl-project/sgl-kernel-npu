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
DEEPEP_VARIANT="deepep"
DEEPEP_IS_A5_BUILD="OFF"

BUILD_ATTENTIONS_MODULE="OFF"
BUILD_DEEPEP_MODULE="OFF"
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
    ./build.sh                                  Build all modules for A3.
    ./build.sh -a deepep [SOC_VERSION]          Build deep_ep; auto-detect A2, A3, or A5.
    ./build.sh -a deepep2 [SOC_VERSION]         Build deep_ep for A2 (compatible alias).
    ./build.sh -a kernels [SOC_VERSION]         Build sgl_kernel_npu.
    ./build.sh -a memory-saver                  Build torch_memory_saver.
    ./build.sh -a attentions                    Build attentions.

TARGET:
    all            (default) Build all modules.
    deepep         Build deep_ep and auto-select ops (A3/A5) or ops2 (A2).
    deepep2        Build deep_ep with ops2 for A2 (compatible alias).
    kernels        Build sgl_kernel_npu only.
    memory-saver   Build torch_memory_saver only.
    attentions     Build attentions only.

SOC_VERSION:
    Ascend910B1         A2 chip. Valid for deepep/deepep2/kernels.
    Ascend910_9382      A3 chip. Valid for all/deepep/kernels.
    Ascend950           A5 chip. Valid for deepep only.
    Ascend950PR_9599    A5 chip. Valid for kernels.
    (omitted)           all: defaults to Ascend910_9382.
                        deepep: auto-detect via npu-smi, fallback to Ascend910_9382.
                        deepep2: defaults to Ascend910B1.
                        kernels: defaults to Ascend910_9382.
                        all: defaults to Ascend910_9382.

Options:
    -d             Enable debug logging.
    -h             Show this help.

Examples:
    ./build.sh                                  # all modules, A3
    ./build.sh -a deepep                        # auto-detect chip
    ./build.sh -a deepep Ascend950              # explicit A5 DeepEP
    ./build.sh -a deepep2                       # A2 DeepEP
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
            BUILD_DEEPEP_MODULE="ON"
            BUILD_KERNELS_MODULE="ON"
            BUILD_MEMORY_SAVER_MODULE="ON"
            DEEPEP_VARIANT="deepep"
            ;;
        deepep )
            BUILD_DEEPEP_MODULE="ON"
            DEEPEP_VARIANT="deepep"
            ;;
        deepep2 )
            BUILD_DEEPEP_MODULE="ON"
            DEEPEP_VARIANT="deepep2"
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
            die "Invalid target '$BUILD_TARGET'. Allowed values: deepep|deepep2|kernels|memory-saver"
            ;;
    esac
}

function detect_deepep_soc_version()
{
    local board_info=""

    # Build containers may not expose an NPU. Preserve the existing A3 default
    # in that case; callers can still provide Ascend950 explicitly.
    if ! command -v npu-smi >/dev/null 2>&1; then
        SOC_VERSION="Ascend910_9382"
        echo "Cannot find npu-smi; defaulting DeepEP SOC_VERSION to $SOC_VERSION"
        return
    fi

    # A5 supports the device-level query and reports Chip Name as Ascend950*.
    board_info="$(npu-smi info -t board -i 0 2>/dev/null || true)"
    if printf '%s\n' "$board_info" |
        grep -Eiq '^[[:space:]]*Chip Name[[:space:]]*:[[:space:]]*Ascend950'; then
        SOC_VERSION="Ascend950"
        echo "Detected A5: DeepEP SOC_VERSION=$SOC_VERSION"
        return
    fi

    # A2 and A3 require a chip ID. Check A2 first because it reports 910B*,
    # while A3 reports Ascend910.
    board_info="$(npu-smi info -t board -i 0 -c 0 2>/dev/null || true)"
    if printf '%s\n' "$board_info" |
        grep -Eiq '^[[:space:]]*Chip Name[[:space:]]*:[[:space:]]*910B'; then
        SOC_VERSION="Ascend910B1"
        echo "Detected A2: DeepEP SOC_VERSION=$SOC_VERSION"
        return
    fi

    if printf '%s\n' "$board_info" |
        grep -Eiq '^[[:space:]]*Chip Name[[:space:]]*:[[:space:]]*Ascend910'; then
        SOC_VERSION="Ascend910_9382"
        echo "Detected A3: DeepEP SOC_VERSION=$SOC_VERSION"
        return
    fi

    die "Cannot determine the device type (A2/A3/A5) from npu-smi output."
}

function configure_soc_version()
{
    case "$BUILD_TARGET" in
        all )
            if [[ -n "$REQUESTED_SOC_VERSION" ]]; then
                die "The default full build does not accept SOC_VERSION; use a specific -a target."
            fi
            SOC_VERSION="Ascend910_9382"
            CMAKE_SOC_VERSION="Ascend910_9382"
            ;;
        deepep )
            if [[ -n "$REQUESTED_SOC_VERSION" ]]; then
                SOC_VERSION="$REQUESTED_SOC_VERSION"
            else
                detect_deepep_soc_version
            fi

            case "$SOC_VERSION" in
                Ascend910B1 )
                    DEEPEP_VARIANT="deepep2"
                    ;;
                Ascend910_9382 | Ascend950 )
                    DEEPEP_VARIANT="deepep"
                    ;;
                * )
                    die "Target 'deepep' supports only Ascend910B1 (A2), Ascend910_9382 (A3), or Ascend950 (A5)."
                    ;;
            esac
            CMAKE_SOC_VERSION="Ascend910_9382"
            ;;
        deepep2 )
            SOC_VERSION="${REQUESTED_SOC_VERSION:-Ascend910B1}"
            if [[ "$SOC_VERSION" != "Ascend910B1" ]]; then
                die "Target 'deepep2' supports only Ascend910B1 (A2)."
            fi
            CMAKE_SOC_VERSION="Ascend910_9382"
            ;;
        kernels )
            SOC_VERSION="${REQUESTED_SOC_VERSION:-Ascend910_9382}"
            if [[ "$SOC_VERSION" == "Ascend950" ]]; then
                die "Target 'kernels' requires an AscendC-supported SoC name instead of the DeepEP alias Ascend950." \
                    "Verified: Ascend950PR_9599"
            fi
            CMAKE_SOC_VERSION="$SOC_VERSION"
            ;;
        memory-saver )
            if [[ -n "$REQUESTED_SOC_VERSION" ]]; then
                die "Target 'memory-saver' does not accept SOC_VERSION."
            fi
            SOC_VERSION=""
            ;;
    esac

    if [[ "$SOC_VERSION" == "Ascend950" ]]; then
        DEEPEP_IS_A5_BUILD="ON"
        export ASCEND_COMPUTE_UNIT="ascend950"
    else
        unset ASCEND_COMPUTE_UNIT
    fi

    echo "Build target: $BUILD_TARGET"
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" ]]; then
        echo "DeepEP variant: $DEEPEP_VARIANT"
        echo "DeepEP SOC_VERSION: $SOC_VERSION"
        echo "DeepEP ASCEND_COMPUTE_UNIT: ${ASCEND_COMPUTE_UNIT:-<unset>}"
    fi
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" || "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        echo "CMake SOC_VERSION: $CMAKE_SOC_VERSION"
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

function prepare_deepep_build()
{
    (
        cd "$PROJECT_ROOT/csrc"
        chmod +x deepep_cmake_build.sh
        chmod +x deepep/build.sh
        chmod +x deepep/compile_ascend_proj.sh

        echo "./deepep_cmake_build.sh all $SOC_VERSION"
        ./deepep_cmake_build.sh all "$SOC_VERSION"

        echo "./deepep/compile_ascend_proj.sh ./deepep $SOC_VERSION $DEEPEP_VARIANT"
        bash ./deepep/compile_ascend_proj.sh \
            ./deepep \
            "$SOC_VERSION" \
            "$DEEPEP_VARIANT"
    )
}

# The top-level CMake project builds two independently controlled components:
# - BUILD_DEEPEP_MODULE=ON: the DeepEP host/pybind adapter (deep_ep_cpp)
# - BUILD_KERNELS_MODULE=ON: the sgl_kernel_npu host library and AscendC kernels
#
# Ascend950 is a DeepEP build alias, but it is not accepted by AscendC's
# host_config.cmake. Keep the top-level CMake SOC on the supported A3 value;
# A5 host differences are enabled separately through DEEPEP_IS_A5_BUILD.
function build_cmake_modules()
{
    local cmake_args=(
        "-DCMAKE_INSTALL_PREFIX=$OUTPUT_DIR"
        "-DASCEND_HOME_PATH=$ASCEND_HOME_PATH"
        "-DASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR"
        "-DSOC_VERSION=$CMAKE_SOC_VERSION"
        "-DDEEPEP_IS_A5_BUILD=$DEEPEP_IS_A5_BUILD"
        "-DBUILD_DEEPEP_MODULE=$BUILD_DEEPEP_MODULE"
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

function build_deepep_kernels()
{
    local kernel_dir=""
    local custom_opp_dir="${PROJECT_ROOT}/python/deep_ep/deep_ep"
    local custom_opp_file=""

    if [[ "$DEEPEP_VARIANT" == "deepep2" ]]; then
        kernel_dir="${PROJECT_ROOT}/csrc/deepep/ops2"
    else
        kernel_dir="${PROJECT_ROOT}/csrc/deepep/ops"
    fi

    (
        cd "$kernel_dir"
        chmod +x build.sh
        ./build.sh

        custom_opp_file="$(find ./build_out \
            -maxdepth 1 \
            -type f \
            -name "custom_opp*.run" \
            -print \
            -quit)"
        if [[ -z "$custom_opp_file" ]]; then
            die "Cannot find custom_opp*.run under $kernel_dir/build_out"
        fi

        echo "Found run package: $custom_opp_file"
        chmod +x "$custom_opp_file"
        rm -rf "$custom_opp_dir/vendors"
        "$custom_opp_file" --install-path="$custom_opp_dir"
    )
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

function make_deepep_package()
{
    (
        cd "$PROJECT_ROOT/python/deep_ep"
        cp -v "$OUTPUT_DIR"/lib/* "$PROJECT_ROOT/python/deep_ep/deep_ep/"
        rm -rf build dist
        python3 setup.py clean --all
        python3 setup.py bdist_wheel
        mv -v dist/deep_ep*.whl "$OUTPUT_DIR/"
        rm -rf dist
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
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" ]]; then
        prepare_deepep_build
    fi
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" || "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        build_cmake_modules
    fi
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" ]]; then
        build_deepep_kernels
    fi
    if [[ "$BUILD_ATTENTIONS_MODULE" == "ON" ]]; then
        build_attentions_kernels
    fi

    ensure_wheel_package

    # Package only the modules selected above.
    if [[ "$BUILD_DEEPEP_MODULE" == "ON" ]]; then
        make_deepep_package
    fi
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
