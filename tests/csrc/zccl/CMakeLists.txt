cmake_minimum_required(VERSION 3.16)
project(ascendc_op_tests)

set(CMAKE_CXX_STANDARD 17)
set(ZCCL_TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(all_gather)
add_subdirectory(reduce_scatter)