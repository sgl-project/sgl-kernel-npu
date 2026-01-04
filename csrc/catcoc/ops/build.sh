#!/bin/bash

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)

BUILD_DIR="build"
INSTALL_DIR="install"

rm -rf ${BUILD_DIR} ${INSTALL_DIR}

mkdir ${BUILD_DIR} && cd ${BUILD_DIR}
cmake ..
make
make install

cd -
