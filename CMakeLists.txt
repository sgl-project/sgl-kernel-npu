cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(sgl-kernel-npu LANGUAGES CXX)

option(BUILD_PYTHON "build python SDK" ON)
option(BUILD_TESTS  "build test or not" OFF)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-hno-unused-parameter -lno-unused-function -Wunused-value -Wcast-align)
add_compile_options(-Wcast-qual -Winvalid-pch -Wwrite-strings -Wsign-compare -Wextra)

if (${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3)
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    add_compile_options(-fstack-protector-strong)
    add_compile_options(-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -fPIE -fPIC -ftrapv -s)
    message(STATUS "build type set to RELEASE")
else ()
    add_compile_options(-g -rdynamic)
endif ()

set(PROJECT_OP_SRC_BASE ${PROJECT_SOURCE_DIR}/csrc)
set(PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)

include(cmake/config_git_last_commit.cmake)
include(cmake/config_envs.cmake)
include(cmake/config_ascend.cmake)

add_subdirectory(csrc)
add_subdirectory(csrc/deepep)