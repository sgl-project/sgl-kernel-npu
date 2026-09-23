cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(fuseep_routing_test LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CATLASS_ARCH 3510 CACHE STRING "3510 for Ascend950, 2201 for A3")
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(ROUTING_INCLUDE_DIR
    "${REPO_ROOT}/csrc/deepep/ops/op_kernel/dispatch_ffn_combine_kernel/moe_init_routing_quant_v2"
    CACHE PATH "FuseEP routing headers to test")

add_executable(test_routing_quant test_routing_quant.cpp)
set_source_files_properties(test_routing_quant.cpp PROPERTIES LANGUAGE ASC)
target_compile_options(test_routing_quant PRIVATE --npu-arch=dav-${CATLASS_ARCH} --cce-auto-sync=off)
target_include_directories(test_routing_quant PRIVATE ${ROUTING_INCLUDE_DIR} $ENV{ASCEND_HOME_PATH}/include)
target_link_directories(test_routing_quant PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(test_routing_quant PRIVATE ascendcl m)
