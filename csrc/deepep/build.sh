#!/bin/bash
export MODULE_NAME="deepep"
export MODULE_SRC_PATH="${SRC_PATH}/${MODULE_NAME}"
export MODULE_SCRIPTS_PATH="${SCRIPTS_PATH}/${MODULE_NAME}"
export MODULE_BUILD_OUT_PATH="${BUILD_OUT_PATH}/${MODULE_NAME}"
export MODULE_TEST_PATH="${TEST_PATH}/${MODULE_NAME}"
IS_EXTRACT=0
SOC_VERSION="all"
ENABLE_UT_BUILD=0
ENABLE_PYBIND_BUILD=0
ENABLE_SRC_BUILD=1

BuildPybind() {
  DIST_OUT_PATH=$MODULE_BUILD_OUT_PATH
  if [ -d $DIST_OUT_PATH/dist ]; then
    rm -rf $DIST_OUT_PATH/dist
  fi
  EXT_PATH=$MODULE_SRC_PATH
  cd $EXT_PATH
  sh build.sh
  DIST_GEN_PATH=$EXT_PATH/dist/
  if [ -d $DIST_GEN_PATH ]; then
    echo "copy $DIST_GEN_PATH to $DIST_OUT_PATH/"
    cp -rf $DIST_GEN_PATH $DIST_OUT_PATH/
  else
    echo $DIST_GEN_PATH does not exist
    echo "BuildPybind fail"
    return 1
  fi
}

BuildTest() {
  cd ${MODULE_TEST_PATH}/ut_gtest
  if [ -d "./build" ]; then
    rm -rf "./build"
  fi
  mkdir ./build
  cd build
  cmake .. && make -j && make install
  if [ $? -ne 0 ]; then
    echo "BuildTest fail"
    return 1
  fi
}

PrintHelp() {
  echo "
./build.sh comm_operator <opt>...
-x   Extract the run package
-c   Target SOC VERSION
     Suport Soc: [ascend910_93, ascend910b4]
-d   Enable debug
-t   enable UT build
-p   enable pybind build
-r   enable code coverage
"
}

while getopts "c:xdtprh" opt; do
  case $opt in
  c)
    SOC_VERSION=$OPTARG
    ;;
  x)
    IS_EXTRACT=1
    ;;
  d)
    export BUILD_TYPE="Debug"
    ;;
  t)
    ENABLE_UT_BUILD=1
    ENABLE_SRC_BUILD=0
    ;;
  p)
    ENABLE_PYBIND_BUILD=1
    ENABLE_SRC_BUILD=0
    ;;
  r)
    export BUILD_TYPE="Debug"
    export ENABLE_COV=1
    ;;
  h)
    PrintHelp
    exit 0
    ;;
  esac
done

echo "Start creating the CMake file"

if [ ! -d "$BUILD_OUT_PATH/${MODULE_NAME}" ]; then
  echo "mkdir $BUILD_OUT_PATH/${MODULE_NAME}"
  mkdir $BUILD_OUT_PATH/${MODULE_NAME}
fi

if [ $ENABLE_SRC_BUILD -eq 1 ]; then
  if [[ "$SOC_VERSION" == "all" ]]; then
    echo "$MODULE_SCRIPTS_PATH/compile_ascend_proj.sh $MODULE_SRC_PATH Ascend910_9382 $IS_EXTRACT $BUILD_TYPE"
    bash $MODULE_SCRIPTS_PATH/compile_ascend_proj.sh $MODULE_SRC_PATH Ascend910_9382 $IS_EXTRACT $BUILD_TYPE
  else
    echo "$MODULE_SCRIPTS_PATH/compile_ascend_proj.sh $MODULE_SRC_PATH $SOC_VERSION $IS_EXTRACT $BUILD_TYPE"
    bash $MODULE_SCRIPTS_PATH/compile_ascend_proj.sh $MODULE_SRC_PATH $SOC_VERSION $IS_EXTRACT $BUILD_TYPE
  fi
  if [ $? -ne 0 ]; then
    exit 1
  fi
fi

if [ $ENABLE_PYBIND_BUILD -eq 1 ]; then
  echo "Start to BuildPybind"
  BuildPybind
  if [ $? -ne 0 ]; then
    exit 1
  fi
fi

if [ $ENABLE_UT_BUILD -eq 1 ]; then
  echo "Start to BuildTest"
  BuildTest
  if [ $? -ne 0 ]; then
    exit 1
  fi
fi
