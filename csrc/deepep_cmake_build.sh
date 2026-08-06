#!/bin/bash
echo "Start the cmake script..."
SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
export ROOT_PATH=$(dirname "$SCRIPT_PATH")
echo ROOT_PATH: $ROOT_PATH
if [ ! -d "./build_out" ]; then
  mkdir build_out
fi
export SRC_PATH="${ROOT_PATH}"
export BUILD_OUT_PATH="${ROOT_PATH}/build_out"
export SCRIPTS_PATH="${ROOT_PATH}"
export TEST_PATH="${ROOT_PATH}/test"

export BUILD_TYPE="${BUILD_TYPE:-Release}"
MODULE_NAME="all"
MODULE_BUILD_ARG=""
IS_MODULE_EXIST=0

function PrintHelp() {
  echo "
./build.sh [module name] <opt>...
If there are no parameters, all modules are compiled in default mode
module list: [deepep]

opt:
-d: Enable debug
-g: Build with symbols using Debug build type
--relwithdebinfo: Build with symbols using RelWithDebInfo
"
}

function ProcessArg() {
  local positional_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
    -d|-g)
      export BUILD_TYPE="Debug"
      shift
      ;;
    --relwithdebinfo)
      export BUILD_TYPE="RelWithDebInfo"
      shift
      ;;
    -h|--help)
      PrintHelp
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        positional_args+=("$1")
        shift
      done
      ;;
    -*)
      echo "unknown flag: $1"
      exit 1
      ;;
    *)
      positional_args+=("$1")
      shift
      ;;
    esac
  done
  set -- "${positional_args[@]}"
}

function IsModuleName() {
  if [ -z "$1" ]; then
    return 1
  fi

  if [[ $1 == -* ]]; then
    return 1
  else
    return 0
  fi
}

if IsModuleName $@; then
  MODULE_NAME=$1
  shift
else
  ProcessArg $@
fi

if [[ "$MODULE_NAME" == "all" || "$MODULE_NAME" == "deepep" ]]; then
    IS_MODULE_EXIST=1
    echo "./deepep/build.sh $@"
    ./deepep/build.sh $@
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

if [ $IS_MODULE_EXIST -eq 0 ]; then
    echo "module not exist"
fi
