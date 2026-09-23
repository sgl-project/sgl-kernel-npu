cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(fuseep_epilogue_test LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CATLASS_ARCH 3510 CACHE STRING "3510 for Ascend950, 2201 for A3")
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CATLASS_INCLUDE_DIR "${REPO_ROOT}/csrc/deepep/ops/third_party/catlass/include"
    CACHE PATH "CATLASS headers from the DeepEP build")
set(EPILOGUE_INCLUDE_DIR
    "${REPO_ROOT}/csrc/deepep/ops/op_kernel/dispatch_ffn_combine_kernel/utils"
    CACHE PATH "FuseEP kernel headers to test")

add_executable(test_epilogue test_epilogue.cpp)
set_source_files_properties(test_epilogue.cpp PROPERTIES LANGUAGE ASC)
target_compile_options(test_epilogue PRIVATE --npu-arch=dav-${CATLASS_ARCH} --cce-auto-sync=off)
target_compile_definitions(test_epilogue PRIVATE CATLASS_ARCH=${CATLASS_ARCH})
target_include_directories(test_epilogue PRIVATE
    ${CATLASS_INCLUDE_DIR} ${EPILOGUE_INCLUDE_DIR} $ENV{ASCEND_HOME_PATH}/include)
target_link_directories(test_epilogue PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(test_epilogue PRIVATE ascendcl)
